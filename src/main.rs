//! IAB taxonomy classifier built on the `agnosticeng/lisa` crates.
//!
//! Runs Laya in two stages and descends a taxonomy tree:
//!   stage 1: one wide multi-choice over all candidates -> top-N shortlist;
//!   stage 2: one small choice over the shortlist.
//! Two batched forward passes per level.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use lisa_engine::models::{
    laya::{Laya, LayaDevice},
    resolve_model_dir,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};

#[derive(Parser)]
#[command(name = "classify", about = "Classify content into an IAB taxonomy with Laya")]
struct Args {
    /// Taxonomy JSON (nested `tree`, as produced for this project)
    #[arg(long, default_value = "data/taxonomy/content-taxonomy-3.1.json")]
    taxonomy: PathBuf,
    /// Input records JSON (`{"inputs":[{id, text, abstract?, ...}]}`)
    #[arg(long, default_value = "data/inputs/articles.json")]
    inputs: PathBuf,
    /// Classify a single record by id
    #[arg(long)]
    id: Option<String>,
    /// Classify every record in the inputs file
    #[arg(long)]
    all: bool,
    /// Classify an ad-hoc text instead of a file record
    #[arg(long)]
    state: Option<String>,
    /// Laya checkpoint: a repo id or a local directory
    #[arg(long, default_value = "convaiinnovations/laya")]
    model: String,
    #[arg(long, value_enum, default_value_t = Device::Metal)]
    device: Device,
    /// Stage-2 option count
    #[arg(long, default_value_t = 8)]
    shortlist: usize,
    /// Option-prompt token budget
    #[arg(long, default_value_t = 512)]
    head_max_len: usize,
    /// Total token budget
    #[arg(long, default_value_t = 8192)]
    max_len: usize,
    /// 1 = top level only, 0 = descend to leaves
    #[arg(long, default_value_t = 1)]
    depth: usize,
    /// Number of stages per level: 1 = wide choice only, 2 = wide + shortlist refine
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    stages: u8,
    /// Bucketed temperature, repeatable, `TYPE:SIZE=TEMP` (e.g. `choice:11+=0.8`)
    #[arg(long = "temperature-by-options", value_name = "TYPE:SIZE=TEMP")]
    temperature_by_options: Vec<String>,
    /// Key the state is passed to Laya under
    #[arg(long, default_value = "article")]
    state_field: String,
    /// Which record field to feed as state: `auto` (abstract then text), `text`, or `abstract`
    #[arg(long, value_parser = ["auto", "text", "abstract"], default_value = "auto")]
    read: String,
    /// Wording used in the question instructions
    #[arg(long, default_value = "web article")]
    subject: String,
    /// Print full JSON results instead of a one-line summary
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Metal,
    Cpu,
}

impl From<Device> for LayaDevice {
    fn from(d: Device) -> Self {
        match d {
            Device::Metal => LayaDevice::Metal,
            Device::Cpu => LayaDevice::Cpu,
        }
    }
}

#[derive(Deserialize)]
struct Taxonomy {
    tree: Vec<Node>,
}

#[derive(Deserialize, Clone)]
struct Node {
    id: String,
    name: String,
    #[serde(default)]
    children: Vec<Node>,
}

#[derive(Deserialize)]
struct Records {
    inputs: Vec<Record>,
}

#[derive(Deserialize, Clone)]
struct Record {
    id: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "abstract")]
    abstract_: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

fn index(tree: &[Node], out: &mut HashMap<String, Node>) {
    for node in tree {
        out.insert(node.id.clone(), node.clone());
        index(&node.children, out);
    }
}

/// Parse `TYPE:SIZE=TEMP` overrides, restricted to the four buckets Laya selects.
fn parse_temp_by_options(items: &[String]) -> Result<Vec<(String, f32)>> {
    const SIZES: [&str; 4] = ["2", "3-5", "6-10", "11+"];
    items
        .iter()
        .map(|s| {
            let (bucket, value) = s.rsplit_once('=').with_context(|| {
                format!("--temperature-by-options wants TYPE:SIZE=TEMP, got {s:?}")
            })?;
            let size = bucket.split_once(':').map(|(_, size)| size).unwrap_or("");
            anyhow::ensure!(
                SIZES.contains(&size),
                "temperature bucket {bucket:?}: size {size:?} must be one of 2, 3-5, 6-10, 11+"
            );
            let temp: f32 = value.trim().parse().with_context(|| format!("invalid temperature {value:?}"))?;
            Ok((bucket.trim().to_string(), temp))
        })
        .collect()
}

/// Stage 1: one wide multi-choice over all candidates, ranked by probability.
fn rank(laya: &Laya, state: &Value, cands: &[Node], subject: &str) -> Result<Vec<(Node, f64)>> {
    let mut criteria = Map::new();
    for c in cands {
        criteria.insert(c.id.clone(), json!(c.name));
    }
    let questions = json!({
        "c:wide": {
            "type": "choice",
            "instructions": format!("Which single category best describes this {subject}? Choose exactly one option."),
            "criteria": Value::Object(criteria),
        }
    });
    let response = laya.system_one(state, &questions)?;
    let probs = &response["answers"]["c:wide"]["probabilities"];
    let mut ranked: Vec<(Node, f64)> = cands
        .iter()
        .map(|c| {
            let p = probs.get(&c.id).and_then(Value::as_f64).unwrap_or(0.0);
            (c.clone(), p)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranked)
}

/// Stage 2: one small choice over the shortlist.
fn choose(laya: &Laya, state: &Value, short: &[(Node, f64)], subject: &str) -> Result<(String, f64, Vec<(String, f64)>)> {
    let mut criteria = Map::new();
    for (c, _) in short {
        criteria.insert(c.id.clone(), json!(c.name));
    }
    let questions = json!({
        "c:shortlist": {
            "type": "choice",
            "instructions": format!("Which single category best describes this {subject}? Choose exactly one option."),
            "criteria": Value::Object(criteria),
        }
    });
    let response = laya.system_one(state, &questions)?;
    let answer = &response["answers"]["c:shortlist"];
    let choice = answer["choice"].as_str().unwrap_or_default().to_string();
    let confidence = answer["probabilities"][&choice].as_f64().unwrap_or(0.0);
    let scored = short.iter().map(|(c, p)| (c.name.clone(), *p)).collect();
    Ok((choice, confidence, scored))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let taxonomy: Taxonomy = serde_json::from_str(
        &fs::read_to_string(&args.taxonomy).with_context(|| format!("reading {}", args.taxonomy.display()))?,
    )
    .context("parsing taxonomy")?;
    let mut nodes = HashMap::new();
    index(&taxonomy.tree, &mut nodes);

    let load_start = Instant::now();
    let dir = resolve_model_dir(&args.model)?;
    let mut laya = Laya::load_device(&dir, args.device.into())?;
    let by_options = parse_temp_by_options(&args.temperature_by_options)?;
    laya.apply_overrides(Some(args.head_max_len), Some(args.max_len), None, &by_options)?;
    eprintln!("[model] loaded in {:.3}s", load_start.elapsed().as_secs_f64());

    let records = if let Some(text) = &args.state {
        vec![Record { id: "inline".into(), text: Some(text.clone()), abstract_: None, url: None }]
    } else {
        let file: Records = serde_json::from_str(
            &fs::read_to_string(&args.inputs).with_context(|| format!("reading {}", args.inputs.display()))?,
        )
        .context("parsing inputs")?;
        file.inputs
    };

    let mut results = Vec::new();
    let run_start = Instant::now();
    let mut ran = 0usize;
    for rec in records {
        if !args.all && args.id.as_deref().is_some_and(|id| id != rec.id) {
            continue;
        }
        let text = match args.read.as_str() {
            "text" => rec.text.clone(),
            "abstract" => rec.abstract_.clone(),
            _ => rec.abstract_.clone().or_else(|| rec.text.clone()),
        }
        .with_context(|| format!("record {} has no {} text", rec.id, args.read))?;
        let state = json!({ args.state_field.clone(): text });

        let rec_start = Instant::now();
        let mut level = taxonomy.tree.clone();
        let mut level_id = "root".to_string();
        let mut path = Vec::new();
        let mut steps = Vec::new();
        let mut q_times: Vec<f64> = Vec::new();
        loop {
            let t = Instant::now();
            let ranked = rank(&laya, &state, &level, &args.subject)?;
            let q1 = t.elapsed().as_secs_f64();
            q_times.push(q1);
            let short: Vec<(Node, f64)> = ranked[..ranked.len().min(args.shortlist)].to_vec();
            let (choice, confidence, shortlist) = if args.stages >= 2 {
                let t = Instant::now();
                let result = choose(&laya, &state, &short, &args.subject)?;
                q_times.push(t.elapsed().as_secs_f64());
                result
            } else {
                let (node, prob) = &short[0];
                (node.id.clone(), *prob, short.iter().map(|(c, p)| (c.name.clone(), *p)).collect())
            };
            let q2 = if args.stages >= 2 {
                format!(" | q2(short, {} opts) {:.3}s", short.len(), q_times.last().copied().unwrap_or(0.0))
            } else {
                String::new()
            };
            eprintln!("  [{}] q1(wide, {} opts) {:.3}s{}", rec.id, level.len(), q1, q2);
            steps.push(json!({ "level": level_id, "shortlist": shortlist, "chosen": choice }));
            let node = nodes.get(&choice).with_context(|| format!("unknown category id {choice}"))?;
            path.push(json!({ "id": node.id, "name": node.name, "confidence": confidence }));
            if args.depth != 0 && path.len() >= args.depth {
                break;
            }
            if node.children.is_empty() {
                break;
            }
            level = node.children.clone();
            level_id = node.id.clone();
        }

        let record_total = rec_start.elapsed().as_secs_f64();
        eprintln!("[{rec_id}] total {record_total:.3}s", rec_id = rec.id);

        let rendered = path
            .iter()
            .map(|p| format!("{} ({:.2})", p["name"].as_str().unwrap_or(""), p["confidence"].as_f64().unwrap_or(0.0)))
            .collect::<Vec<_>>()
            .join(" > ");
        println!("{}: {}", rec.id, rendered);

        let mut result = json!({
            "id": rec.id,
            "taxonomy": args.taxonomy,
            "path": path,
            "steps": steps,
            "timings": { "questions_s": q_times, "total_s": record_total },
        });
        if let Some(url) = rec.url {
            result["url"] = json!(url);
        }
        results.push(result);
        ran += 1;
    }

    if ran > 0 {
        eprintln!("[{ran} record(s)] classify {:.3}s", run_start.elapsed().as_secs_f64());
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}
