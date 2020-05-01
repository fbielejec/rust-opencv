use opencv::{
    prelude::*,
    core::{Vec3b, Scalar},
};

/*
 * https://docs.opencv.org/master/d6/d6d/tutorial_mat_the_basic_image_container.html
 */
pub fn run () -> opencv::Result<()> {

    // create a matrix
    // <opencv::core::Vec3<u8> as Trait>::typ
    let m = Mat::new_rows_cols_with_default(2, 2,
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
    let pixel = m.at_2d::<Vec3b>(0, 0)?;
    println!("{:#?}", pixel);

    let points = m.at::<Vec3b>(0)?;
    println!("- POINTS - {:#?}", points);
    println!("- POINT - {:#?}", points [0]);

    // construct Mat using a literal
    let mat = Mat::from_slice_2d(&[
        &[ 1,  2,  3,  4],
        &[ 5,  6,  7,  8],
        &[ 9, 10, 11, 12],
        &[13, 14, 15, 16u8],
    ])?;

    println!("DATA : {:#?}", mat.data ()?);

    Ok(())
}
