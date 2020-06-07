use {
    opencv::{
        core::{self, psnr},
        objdetect,
        imgproc,
        types,
        prelude::*,
        videoio::{self, VideoCapture},
        highgui
    },
    crate::{utils}
};

/*
 * Track pupils movements
 * Move cursor
 */
pub fn run () -> opencv::Result<()> {

    let name : &str = "/opt/opencv/opencv-4.2.0/data/haarcascades/haarcascade_frontalface_alt.xml";

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let (xml, mut cam) = {
        (
            core::find_file(name, true, false)?,
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?,  // 0 is the default camera
        )
    };

    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    let mut face : objdetect::CascadeClassifier = objdetect::CascadeClassifier::new(&xml)?;

    loop {

        let mut frame = Mat::default()?;
        cam.read(&mut frame)?;

        highgui::imshow(window, &frame)?;

        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}
