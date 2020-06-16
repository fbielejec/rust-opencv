extern crate opencv;

use opencv::{
    prelude::*,
    // Result,
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
