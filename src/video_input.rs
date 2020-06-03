// extern crate opencv;

// use opencv::core as cv;

use {
    opencv::{
        core::{self, psnr},
        prelude::*,
        // videoio,
        videoio::{self, VideoCapture},
        highgui,
    },
    crate::{utils}
};

/*
 * This program shows how to read a video file with OpenCV and compare two video files frame-by-frame using
 * PSNR:  https://docs.rs/opencv/0.33.0/opencv/core/fn.psnr.html
 * https://docs.opencv.org/master/d5/dc4/tutorial_video_input_psnr_ssim.html
 * https://github.com/twistedfall/opencv-rust/blob/master/examples/video_capture.rs
 */
pub fn run () -> opencv::Result<()> {

    const PSNR_TRIGGER_VALUE: f64 = 40.0;

    let mut capture_reference = VideoCapture::from_file("Megamind.avi", videoio::CAP_ANY)?;
    if !VideoCapture::is_opened(&capture_reference)? {
        panic!("Unable to open reference video!");
    }

    let mut capture_test = VideoCapture::from_file("Megamind_bugy.avi", videoio::CAP_ANY)?;
    if !VideoCapture::is_opened(&capture_test)? {
        panic!("Unable to open test video!");
    }

    let mut frame_reference = Mat::default()?;
    let mut frame_test = Mat::default()?;

    let mut frame_number = 0;
    loop {

        // reads frames one by one
        capture_reference.read(&mut frame_reference)?;
        capture_test.read(&mut frame_test)?;

        if( utils::empty(&frame_reference)? || utils::empty(&frame_test)? ) {
            println!("No more video frames to read");
            break;
        }

        // result values are anywhere between 30 and 50 for video compression, where higher is better. If the images significantly differ you'll get much lower ones ~15.
        let psnr_value = psnr(&frame_reference, &frame_test, 255.0)?;

        println!("Frame: #{}", frame_number);
        println!("{} dB", psnr_value);

        if (psnr_value < PSNR_TRIGGER_VALUE) {
            // let mssim_value = getMSSIM(frameReference, frameUnderTest);
            // cout << " MSSIM: "
            //     << " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
            //     << " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
            //     << " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";
        }

        frame_number = frame_number + 1;
    }

    Ok(())
}
