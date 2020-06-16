// extern crate opencv;

use opencv::core as cv;

use opencv::{
    prelude::*,
    core::{ Scalar, Vec3b },
    imgcodecs
};

use crate::{utils};

/*
 * Demonstrates how to scan and reduce an image
 * https://docs.opencv.org/master/db/da5/tutorial_how_to_scan_images.html
 * https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/core/how_to_scan_images/how_to_scan_images.cpp
 */
pub fn run () -> opencv::Result<()> {

    fn scan_image_and_reduce (m : &mut Mat, table : [u8; 256]) -> &Mat {
        if m.depth ().unwrap () != u8::depth() {
            panic!("Only char type matrices!");
        }

        let n_channels = m.channels ().unwrap ();
        let n_rows = m.rows ();
        let mut n_cols = m.cols ();
        // let mut channels = m.channels ().unwrap ();

        // continuous matrix can be accessed by col number:
        // let pixel = m.at_mut::<Vec3b>(262143).unwrap ();

        // or by row, col:
        // let pixel = m.at_2d::<Vec3b>(511, 511).unwrap ();
        // println!("rows: {} cols: {} elem-size: {}", n_rows, n_cols, channels);
        // println!("pixel [BGR]: {:#?}", pixel);
        // println!("pixel [B] {:#?}", pixel [0]);
        // println!("reduced pixel [B] {:#?}", table [pixel [0] as usize]);

        if m.is_continuous ().unwrap () {

            n_cols *= n_rows;
            // n_rows = 1;

            for i in 0..n_cols {
                let pixel = m.at_mut::<Vec3b>(i).unwrap ();
                for j in 0..n_channels as usize {
                    pixel [j] = table [pixel [j] as usize];
                }
            }

        }

        m
    }

    // println!("{} {}", imgcodecs::IMREAD_COLOR, imgcodecs::IMREAD_GRAYSCALE);

    let image : Mat = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_COLOR)?;

    // get an element
    // let row = image.at_2d::<Vec3b>(0, 0)?;
    // println!("{:#?}", row);

    // create a lookup table
    let divide_with: u8 = 100;
    let mut table: [u8; 256] = [0; 256];
    (0..256).for_each (|i: usize| {
        table [i] = divide_with * (i as u8 / divide_with);
    });

    println!("{} {}", table [0], table [255]);

    let mut image_clone: Mat = image.clone()?;
    let image_reduced = scan_image_and_reduce(&mut image_clone, table);

    utils::display_img(&image_reduced, None).expect ("Could not display image");

    // LUT
    // http://www.poumeyrol.fr/doc/opencv-rust/opencv/core/fn.lut.html
    let image_clone = image.clone()?;
    let look_up_table : Mat = Mat::from_slice (&table)?;

    // println!("{:#?}", look_up_table);
    // println!("{} {}", look_up_table.at_2d::<u8> (0,0)?, look_up_table.at_2d::<u8> (0,255)?);

    let mut image_reduced : Mat = Mat::new_rows_cols_with_default(image.rows (), image.cols (), Vec3b::typ(), Scalar::default())?;

    // ok() converts Result<T,E> to Option<T>, converting errors to None (ingnoring None does not trigger a warning).
    cv::lut(&image_clone, &look_up_table, &mut image_reduced).ok();
    utils::display_img(&image_reduced, None).ok();

    // display_img(&image);

    Ok(())
}
