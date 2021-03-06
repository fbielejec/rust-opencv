extern crate opencv;
extern crate enigo;

mod utils;
// mod mouse;
mod mat_the_basic_image_container;
mod how_to_scan_images;
mod mat_mask_operations;
mod mat_operations;
mod video_input;
mod cascade_classifier;
mod timm_barth;
mod eye_tracker;
mod watermark_detect;

fn main() {

    // let mut input = String::new();

    println!("OpenCV tutorials:");
    println!("1) mat_the_basic_image_container");
    println!("2) how_to_scan_images");
    println!("3) mat_mask_operations");
    println!("4) mat_operations");
    println!("5) video_input");
    println!("6) cascade_classifier");
    println!("7) eye_tracker");
    println!("8) watermark_detect");

    // io::stdin().read_line(&mut input).expect ("Failed to read the line");

    let number: u32 = 8;
    // match input.trim().parse() {
    //     Ok(num) => num,
    //     Err(_) => panic!("Aaaa!!!"),
    // };

    match number {
        1 => mat_the_basic_image_container::run ().expect("Runtime error"),
        2 => how_to_scan_images::run ().expect("Runtime error"),
        3 => mat_mask_operations::run ().expect("Runtime error"),
        4 => mat_operations::run ().expect("Runtime error"),
        5 => video_input::run ().expect("Runtime error"),
        6 => cascade_classifier::run ().expect("Runtime error"),
        7 => eye_tracker::run ().expect("Runtime error"),
        8 => watermark_detect::run ().expect("Runtime error"),

        _ => println!("meh"),
    }

}
