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

    let points = m.at::<Vec3b>(0)?;
    println!("- POINTS - {:#?}", points);
    println!("- POINT - {:#?}", points [0]);

    // construct Mat using a literal
    let mut mat = Mat::from_slice_2d(&[
        &[ 1,  2,  3,  4],
        &[ 5,  6,  7,  8],
        &[ 9, 10, 11, 12],
        &[13, 14, 15, 16u8],
    ])?;

    // println!("DATA : {:#?}", mat.data ().unwrap ());


    Ok(())
}

/*
 * Demonstrates how to scan and reduce an image
 * https://docs.opencv.org/master/db/da5/tutorial_how_to_scan_images.html
 * https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/core/how_to_scan_images/how_to_scan_images.cpp
 */
// TODO : reduce w. the table
fn scan_images () -> opencv::Result<()> {

    fn scan_image_and_reduce (m : &mut Mat, table : [u8; 256] ) -> &Mat {
        if (m.depth ().unwrap () != u8::depth()) {
            panic!("Only char type matrices!");
        }

        let channels = m.channels ().unwrap ();
        let mut n_rows = m.rows ();
        let mut n_cols = m.cols ();

        if m.is_continuous ().unwrap () {
            n_cols *= n_rows;
            n_rows = 1;

        }

        let points = m.at_mut::<Vec3b>(0).unwrap ();
        println!("{:#?}", points [0]);
        // points [0] = 121;
        // println!("{:#?}", points [0]);



        m
    }

    let divide_with: u8 = 10;

    // println!("{} {}", imgcodecs::IMREAD_COLOR, imgcodecs::IMREAD_GRAYSCALE);

    let mut image : Mat = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_COLOR)?;

    println!("{:#?}", image);

    // get an element
    let row = image.at_2d::<Vec3b>(0, 0)?;
    println!("{:#?}", row);

    // create a lookup table
    let mut table: [u8; 256] = [0; 256];
    (0..256).for_each (|i: usize| {
        table [i] = divide_with * (i as u8 / divide_with);
    });

    println!("{} {}", table [0], table [255]);

    let mut image_clone: Mat = image.clone()?;
    let image_reduced = scan_image_and_reduce(&mut image_clone, table);


    Ok(())
}


fn main() {
    // hello_opencv().unwrap()
    mat ().unwrap ();
    scan_images ().unwrap ();
}
