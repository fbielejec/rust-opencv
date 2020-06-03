extern crate opencv;

use opencv::{
    prelude::*,
    // Result,
    highgui
};

pub fn display_img(image: &Mat) -> opencv::Result<()> {
    highgui::named_window("hello opencv!", 0)?;
    highgui::imshow("hello opencv!", image)?;
    highgui::wait_key(10000)?;

    Ok(())
}

pub fn empty(mat: &Mat) -> opencv::Result<bool> {
    let size = mat.size()?;
    Ok(size.width == 0 && size.height == 0)
}
