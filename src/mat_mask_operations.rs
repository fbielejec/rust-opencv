use opencv::{
    core::{self, Point, Scalar, Vec3b},
    imgproc,
    prelude::*,
    imgcodecs
};

use crate::{utils};

#[allow(dead_code)]
fn saturate_cast (value: &i32) -> u8 {
    (u8::min_value () as i32 +
     (u8::max_value () as i32 / i32::max_value () ) * value) as u8
}

fn sharpen (image: &Mat, result: &mut Mat) -> opencv::Result<()> {

    let n_channels: i32 = image.channels ()? ;
    let n_rows: i32 = image.rows () ;
    let n_cols: i32 = image.cols () ;

    // for i in 1..n_rows - 1 {
    //     for j in 1..n_cols - 1 {

    //         let top = image.at_2d::<Vec3b>(i + 1, j)?;
    //         let bottom = image.at_2d::<Vec3b>(i - 1, j)?;
    //         let current = image.at_2d::<Vec3b>(i, j)?;
    //         let left = image.at_2d::<Vec3b>(i, j - 1)?;
    //         let right = image.at_2d::<Vec3b>(i, j + 1)?;

    //         let pixel = result.at_2d_mut::<Vec3b>(i, j)?;
    //         for k in 0..n_channels as usize {

    //             let value : i32 = 5 * current [k] as i32 - top [k] as i32
    //                 - bottom [k] as i32 - left [k] as i32 - right [k] as i32;

    //             pixel [k] = saturate_cast (&value);

    //         }
    //     }
    // }

    // TODO : broken

    let n: i32 = n_rows * n_cols;
    for j in 3..(n - 3) {

        let top = image.at::<Vec3b>(j - 3)?;
        let bottom = image.at::<Vec3b>(j + 3)?;
        let current = image.at::<Vec3b>(j)?;
        let left = image.at::<Vec3b>(j - 1)?;
        let right = image.at::<Vec3b>(j + 1)?;

        let pixel = result.at_mut::<Vec3b>(j)?;

        for k in 0..n_channels as usize {
            let ( value, _) = 5u8.overflowing_mul(current[k]);
            let ( value, _) = value.overflowing_sub(top[k]);
            let ( value, _) = value.overflowing_sub(bottom[k]);
            let ( value, _) = value.overflowing_sub(left[k]);
            let ( value, _) = value.overflowing_sub(right[k]);
            pixel [k] = value;
        }
    }

    Ok(())
}

/*
 * Demonstrates how to apply a (sharpen) mask to an image
 * https://docs.opencv.org/master/d7/d37/tutorial_mat_mask_operations.html
 */
pub fn run () -> opencv::Result<()> {

    let image : Mat = imgcodecs::imread("lena.jpg", imgcodecs::IMREAD_COLOR)?;
    let mut image_sharpened : Mat = Mat::new_rows_cols_with_default(image.rows (), image.cols (), image.typ ()?, Scalar::default())?;
    sharpen (&image, &mut image_sharpened).ok();

    utils::display_img(&image_sharpened, None).expect ("Error displaying image");

    // filter2D
    // https://docs.rs/opencv/0.33.0/opencv/imgproc/fn.filter_2d.html

    let kernel = Mat::from_slice_2d(&[
        &[ 0,  -1,  0],
        &[ -1,  5,  -1],
        &[ 0, -1, 0i32]
    ])?;

    let mut image_sharpened : Mat = Mat::new_rows_cols_with_default(image.rows (), image.cols (), image.typ ()?, Scalar::default())?;
    imgproc::filter_2d( &image, &mut image_sharpened, image.depth()?, &kernel, Point::new(-1,-1), 0.0, core::BORDER_DEFAULT).ok();

    utils::display_img(&image_sharpened, None).ok();

    Ok(())
}
