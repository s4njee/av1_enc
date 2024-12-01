use std::path::Path;
use std::process::{Command, Stdio};
use walkdir::WalkDir;
use std::error::Error;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use clap::Parser;
use rayon::ThreadPoolBuilder;
use num_cpus;
use std::fs;
use std::io::{BufRead, BufReader};
use regex::Regex;

#[derive(Parser)]
#[command(name = "av1_enc")]
#[command(about = "Converts MP4 files to AV1 format using SVT-AV1")]
struct Args {
    /// Base directory to scan for video files
    #[arg(default_value = ".")]
    base_dir: String,

    /// SVT-AV1 preset (0-13, lower is better quality but slower)
    #[arg(short, long, default_value = "8")]
    preset: u32,

    /// Constant Rate Factor (0-63, lower is better quality)
    #[arg(short = 'c', long, default_value = "30")]
    crf: u32,

    /// Tune parameter (default: 0)
    /// 0=vq, 1=psnr, 2=ssim
    #[arg(short = 't', long, default_value = "0")]
    tune: u32,

    /// Video filter string (e.g., "scale=1920:1080")
    #[arg(short = 'v', long)]
    vf: Option<String>,

   /// Audio bitrate in kbps (e.g., 128)
    #[arg(short = 'b', long)]
    audio_bitrate: Option<u32>,

    /// Copy audio stream instead of re-encoding
    #[arg(short = 'k', long)]
    copy_audio: bool,

    /// Number of parallel videos to process (default: number of CPU cores)
    #[arg(short = 'j', long, default_value_t = num_cpus::get())]
    threads: usize,

    /// Delete original files after successful conversion
    #[arg(short = 'd', long)]
    delete_originals: bool,
}

struct VideoInfo {
    path: String,
    frame_count: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Validate preset range
    if args.preset > 13 {
        eprintln!("Error: Preset must be between 0 and 13");
        std::process::exit(1);
    }

    // Validate CRF range
    if args.crf > 63 {
        eprintln!("Error: CRF must be between 0 and 63");
        std::process::exit(1);
    }

    // Validate tune range
    if args.tune > 2 {
        eprintln!("Error: Tune must be between 0 and 2");
        std::process::exit(1);
    }

    // Validate thread count
    if args.threads < 1 {
        eprintln!("Error: Thread count must be at least 1");
        std::process::exit(1);
    }

    // Check dependencies
    if !is_ffmpeg_installed() {
        eprintln!("Error: ffmpeg is not installed. Please install it first.");
        std::process::exit(1);
    }
    if !is_ffprobe_installed() {
        eprintln!("Error: ffprobe is not installed. Please install it first.");
        std::process::exit(1);
    }

    // Configure thread pool
    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    // Check if base directory exists
    if !Path::new(&args.base_dir).exists() {
        eprintln!("Error: Directory '{}' does not exist", args.base_dir);
        std::process::exit(1);
    }

    println!("Starting AV1 encoding...");
    println!("Base directory: {}", args.base_dir);
    println!("SVT-AV1 preset: {}", args.preset);
    println!("CRF: {}", args.crf);
    println!("Tune: {} ({})", args.tune, match args.tune {
        0 => "vq",
        1 => "psnr",
        2 => "ssim",
        _ => "unknown"
    });
    if let Some(ref vf) = args.vf {
        println!("Video filter: {}", vf);
    }
    if args.copy_audio {
        println!("Audio: Copy original streams");
    } else {
        println!("Audio: Opus encode{}", args.audio_bitrate.map_or(String::new(), |br| format!(" @ {}k", br)));
    }
    println!("Using {} parallel video processes", args.threads);
    if args.delete_originals {
        println!("Original files will be deleted after conversion");
    }
    let start_time = std::time::Instant::now();

    // Collect all video files and their frame counts
    println!("Scanning for MP4 files and counting frames...");
    let mut video_files: Vec<VideoInfo> = Vec::new();
    let frame_scanning_pb = ProgressBar::new_spinner();
    frame_scanning_pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}").unwrap());
    
    for entry in WalkDir::new(&args.base_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .map(|ext| ext.to_ascii_lowercase() == "mp4")
                .unwrap_or(false)
        }) {
            frame_scanning_pb.set_message(format!("Scanning: {}", entry.path().display()));
            if let Ok(frame_count) = get_frame_count(entry.path()) {
                video_files.push(VideoInfo {
                    path: entry.path().to_string_lossy().into_owned(),
                    frame_count,
                });
            }
    }
    frame_scanning_pb.finish_and_clear();

    let total_files = video_files.len();
    if total_files == 0 {
        println!("No MP4 files found in directory: {}", args.base_dir);
        return Ok(());
    }

    let total_frames: usize = video_files.iter().map(|v| v.frame_count).sum();
    println!("Found {} files with {} total frames to encode", total_files, total_frames);

    // Setup progress bar for total frames
    let pb = ProgressBar::new(total_frames as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames ({percent}%) {msg}")
        .unwrap()
        .progress_chars("#>-"));

    // Counters
    let successful_conversions = Arc::new(AtomicUsize::new(0));
    let successful_frames = Arc::new(AtomicUsize::new(0));
    let deleted_files = Arc::new(AtomicUsize::new(0));
    let preset = args.preset;
    let should_delete = args.delete_originals;

    // Process files in parallel
    video_files.par_iter()
        .for_each(|video| {
            match convert_to_av1(
                &video.path, 
                preset,
                args.crf,
                args.tune,
                args.copy_audio,
		args.audio_bitrate,
                args.vf.as_deref(),
                &pb, 
                successful_frames.clone()
            ) {
                Ok(_) => {
                    successful_conversions.fetch_add(1, Ordering::Relaxed);
                    
                    if should_delete {
                        match fs::remove_file(&video.path) {
                            Ok(_) => {
                                deleted_files.fetch_add(1, Ordering::Relaxed);
                                pb.set_message(format!("Converted and deleted: {}", video.path));
                            },
                            Err(e) => {
                                eprintln!("Failed to delete {}: {}", video.path, e);
                                pb.set_message(format!("Converted (delete failed): {}", video.path));
                            }
                        }
                    } else {
                        pb.set_message(format!("Converted: {}", video.path));
                    }
                },
                Err(e) => {
                    eprintln!("Failed to convert {}: {}", video.path, e);
                }
            }
        });
    pb.finish_with_message("Encoding complete!");

    let converted_files = successful_conversions.load(Ordering::Relaxed);
    let converted_frames = successful_frames.load(Ordering::Relaxed);
    let deleted_count = deleted_files.load(Ordering::Relaxed);
    let duration = start_time.elapsed();

    println!("\nEncoding Summary:");
    println!("Total files processed: {}", total_files);
    println!("Successfully converted: {}", converted_files);
    println!("Failed conversions: {}", total_files - converted_files);
    println!("Total frames encoded: {}/{}", converted_frames, total_frames);
    if args.delete_originals {
        println!("Original files deleted: {}", deleted_count);
    }
    println!("Time taken: {:.2} seconds", duration.as_secs_f64());

    Ok(())
}

fn convert_to_av1(
    input_path: &str,
    preset: u32,
    crf: u32,
    tune: u32,
    copy_audio: bool,
    audio_bitrate: Option<u32>,
    vf: Option<&str>,
    pb: &ProgressBar,
    frame_counter: Arc<AtomicUsize>
) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(input_path).with_extension("mkv");
    
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-i")
        .arg(input_path)
        .arg("-map_metadata")  // Copy all metadata from input
        .arg("0")
        .arg("-c:v")
        .arg("libsvtav1")
        .arg("-preset")
        .arg(preset.to_string())
        .arg("-crf")
        .arg(crf.to_string())
        .arg("-svtav1-params")
        .arg(format!("tune={}", tune));

    // Add video filter if specified
    if let Some(filter) = vf {
        cmd.arg("-vf").arg(filter);
    }
    if copy_audio {
        cmd.arg("-c:a").arg("copy");
    } else {
        cmd.arg("-c:a").arg("libopus");
        if let Some(bitrate) = audio_bitrate {
            cmd.arg("-b:a").arg(format!("{}k", bitrate));
        }
    }

        cmd.arg("-progress")
        .arg("pipe:1")
        .arg("-y")
        .arg(&output_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut process = cmd.spawn()?;

    let frame_regex = Regex::new(r"frame=\s*(\d+)")?;
    let stdout = process.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    let mut last_frame = 0;
    for line in reader.lines() {
        if let Ok(line) = line {
            if let Some(caps) = frame_regex.captures(&line) {
                if let Some(frame_str) = caps.get(1) {
                    if let Ok(current_frame) = frame_str.as_str().parse::<usize>() {
                        let frame_diff = current_frame - last_frame;
                        frame_counter.fetch_add(frame_diff, Ordering::Relaxed);
                        pb.inc(frame_diff as u64);
                        last_frame = current_frame;
                    }
                }
            }
        }
    }

    let status = process.wait()?;
    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }

    Ok(())
}

fn get_frame_count(path: &Path) -> Result<usize, Box<dyn Error>> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-count_packets")
        .arg("-show_entries")
        .arg("stream=nb_read_packets")
        .arg("-of")
        .arg("csv=p=0")
        .arg(path)
        .output()?;

    let frame_count = String::from_utf8(output.stdout)?
        .trim()
        .parse::<usize>()?;

    Ok(frame_count)
}

fn is_ffmpeg_installed() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn is_ffprobe_installed() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
