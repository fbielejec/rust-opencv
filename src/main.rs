extern crate opencv;
use std::io;

mod utils;
mod mat_the_basic_image_container;
mod how_to_scan_images;
mod mat_mask_operations;
mod mat_operations;
mod video_input;

fn main() {

    let mut input = String::new();

    println!("OpenCV tutorials:");
    println!("1) mat_the_basic_image_container");
    println!("2) how_to_scan_images");
    println!("3) mat_mask_operations");
    println!("4) mat_operations");
    println!("5) video_input");

    // io::stdin().read_line(&mut input).expect ("Failed to read the line");

    let number: u32 = 5;
        // match input.trim().parse() {
        //     Ok(num) => num,
        //     Err(_) => panic!("Aaaa!!!"),
        // };

    // TODO : video
    // https://docs.opencv.org/master/df/d2c/tutorial_table_of_content_videoio.html

    match number {
        1 => mat_the_basic_image_container::run ().expect("Runtime error"),
        2 => how_to_scan_images::run ().expect("Runtime error"),
        3 => mat_mask_operations::run ().expect("Runtime error"),
        4 => mat_operations::run ().expect("Runtime error"),
        5 => video_input::run ().expect("Runtime error"),
        _ => println!("meh"),
    }

}
