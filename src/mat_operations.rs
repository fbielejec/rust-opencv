use opencv::{
    prelude::*,
    core::{ Vec3b },
    imgcodecs
};

/*
 * https://docs.opencv.org/master/d5/d98/tutorial_mat_operations.html
 * Demonstrates Basic operations with images
 */
pub fn run () -> opencv::Result<()> {

    let image : Mat = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_COLOR)?;

    // Accessing pixel intensity values
    let pixel : &Vec3b = image.at_2d::<Vec3b>(0, 0)?;

    let blue: u8 = pixel[0];
    let green: u8 = pixel[1];
    let red: u8 = pixel[2];

    println!("R : {} G : {} B : {}", red, green, blue);

    // TODO

    Ok(())
}
