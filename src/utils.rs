extern crate opencv;

use opencv::{
    prelude::*,
    imgproc,
    highgui
};

pub fn display_img(image: &Mat, title: Option<&str>) -> opencv::Result<()> {
    let t = title.unwrap_or ("window");
    highgui::named_window(t, 0)?;
    highgui::imshow(t, image)?;
    highgui::wait_key(10000)?;

    Ok(())
}

pub fn empty(mat: &Mat) -> opencv::Result<bool> {
    let size = mat.size()?;
    Ok(size.width == 0 && size.height == 0)
}

pub fn enhance_frame (frame : &Mat)
                  -> opencv::Result<Mat>{
    let mut gray = Mat::default()?;
    let mut equalized = Mat::default()?;

    imgproc::cvt_color(
        &frame,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0
    )?;

    imgproc::equalize_hist (&gray,
                            &mut equalized)?;

    Ok(equalized)
}
