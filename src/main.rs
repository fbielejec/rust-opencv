extern crate opencv;
use opencv::core as cv;

use opencv::{
    core::{self, MatConstIterator, Point, Rect, Scalar, Size, Vec3, Vec2b, Vec3b, Vec3d, Vec3f, Vec4w},
    Error,
    prelude::*,
    Result,
    types::{VectorOfi32, VectorOfMat},
    highgui,
    imgcodecs
};

// https://docs.opencv.org/master/d6/d6d/tutorial_mat_the_basic_image_container.html
// https://github.com/twistedfall/opencv-rust/blob/master/tests/mat.rs

fn hello_opencv() -> opencv::Result<()> {
    let image = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_GRAYSCALE).unwrap();
    highgui::named_window("hello opencv!", 0).unwrap();
    highgui::imshow("hello opencv!", &image).unwrap();
    highgui::wait_key(10000).unwrap();

    Ok(())
}

/*
 * https://docs.opencv.org/master/d6/d6d/tutorial_mat_the_basic_image_container.html
 */
fn mat () -> opencv::Result<()> {

    // create a matrix
    // <opencv::core::Vec3<u8> as Trait>::typ
    let mut m = Mat::new_rows_cols_with_default(2, 2,
                                                Vec3b::typ(),
                                                // u8::typ(),
                                                Scalar::new(0., 0., 0., 255.))?;

    println!("{:#?}", m);

    // type: "CV_8UC1",
    println!("{:#?}", m.typ ());

    // get a row
    let row = m.at_row::<Vec3b>(0)?;
    println!("{:#?}", row);

    // get an element
    let row = m.at_2d::<Vec3b>(0, 0)?;
    println!("{:#?}", row);

    // literal
    let mut mat = Mat::from_slice_2d(&[
        &[ 1,  2,  3,  4],
        &[ 5,  6,  7,  8],
        &[ 9, 10, 11, 12],
        &[13, 14, 15, 16u8],
    ])?;

    println!("{:#?}", m);

    Ok(())
}

// https://docs.opencv.org/master/db/da5/tutorial_how_to_scan_images.html
fn scan_images () -> opencv::Result<()> {

    // println!("{} {}", imgcodecs::IMREAD_COLOR, imgcodecs::IMREAD_GRAYSCALE);

    let mut image : Mat = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_COLOR)?;

    println!("{:#?}", image);

    // get an element
    let row = image.at_2d::<Vec3b>(0, 0)?;
    println!("{:#?}", row);

    // TODO create a lookup table

    // uchar table[256];
    // for (int i = 0; i < 256; ++i)
    //     table[i] = (uchar)(divideWith * (i/divideWith));




    Ok(())
}


fn main() {
    // hello_opencv().unwrap()
    mat ().unwrap ();
    scan_images ().unwrap ();
}
