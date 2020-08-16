use {
    opencv::{
        core::{self, Point, Scalar, CV_64F, Rect, Vec3b, CV_32F, NormTypes },
        prelude::*,
        videoio::{self, VideoCapture},
        imgcodecs,
        highgui,
        imgproc::{self, TemplateMatchModes}
    },
    crate::{ utils }
};

pub fn run () -> opencv::Result<()> {

    let template_match_mode: TemplateMatchModes = TemplateMatchModes::TM_SQDIFF;
    let norm_type: NormTypes = NormTypes::NORM_L2;

    // let mut template : Mat = imgcodecs::imread("watermark.jpg", imgcodecs::IMREAD_COLOR)?;
    // let mut template : Mat = imgcodecs::imread("watermark2.jpg", imgcodecs::IMREAD_COLOR)?;
    // let mut template : Mat = imgcodecs::imread("watermark3.png", imgcodecs::IMREAD_COLOR)?;
    let mut template : Mat = imgcodecs::imread("watermark4.png", imgcodecs::IMREAD_COLOR)?;
    // let mut video = VideoCapture::from_file("tiktok.mp4", videoio::CAP_ANY)?;
    let mut video = VideoCapture::from_file("/home/filip/Downloads/download.mp4", videoio::CAP_ANY)?;

    let window_name = "test";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    // let mut template = utils::enhance_frame (&template)?;

    let mut frame = Mat::default()?;
    loop {

        // reads frames one by one
        video.read(&mut frame)?;
        // frame = utils::enhance_frame (&frame)?;

        if utils::empty(&frame)? {
            println!("No more video frames to read");
            break;
        }

        let mut result : Mat = Mat::new_rows_cols_with_default(frame.rows (),
                                                               frame.cols (),
                                                               f32::typ (),
                                                               Scalar::default ())?;

        let match_mode: i32 = match template_match_mode {
            TemplateMatchModes::TM_SQDIFF => 0,
            TemplateMatchModes::TM_SQDIFF_NORMED => 1,
            TemplateMatchModes::TM_CCORR => 2,
            TemplateMatchModes::TM_CCORR_NORMED => 3,
            TemplateMatchModes::TM_CCOEFF => 4,
            TemplateMatchModes::TM_CCOEFF_NORMED => 5
        };

        imgproc::match_template(
            &mut frame,
            &mut template,
            &mut result,
            match_mode,
            &core::no_array ()? // mask
        )?;

        // println!("result {:#!}", result);

        let mut normalized_result : Mat = Mat::new_rows_cols_with_default(result.rows (),
                                                                          result.cols (),
                                                                          f32::typ (),
                                                                          Scalar::default())?;

        let norm: i32 = match norm_type {
            NormTypes::NORM_INF => 0,
            NormTypes::NORM_L1 => 1,
            NormTypes::NORM_L2 => 2,
            NormTypes::NORM_L2SQR => 3,
            NormTypes::NORM_HAMMING => 4,
            NormTypes::NORM_HAMMING2 => 5,
            NormTypes::NORM_RELATIVE => 6,
            NormTypes::NORM_MINMAX => 7
        };

        core::normalize(
            &result,
            &mut normalized_result,
            1.0, // alpha
            0.0, // beta
            norm, // NORM_MINMAX
            -1,
            &core::no_array ()? // mask
        )?;

        // find the optimum
        let mut min_point = Point::default ();
        let mut max_point = Point::default ();

        let mut min_value : f64 = 0.0;
        let mut max_value : f64 = 0.0;

        core::min_max_loc(
            &normalized_result,
            // &result,
            &mut min_value,
            &mut max_value,
            &mut min_point, // NULL
            &mut max_point,
            &core::no_array ()?
        )?;

        let mut match_location = Point::default ();
        if ( template_match_mode == TemplateMatchModes::TM_SQDIFF ||
             template_match_mode == TemplateMatchModes::TM_SQDIFF_NORMED ) {
            match_location = min_point;
        } else {
            match_location = max_point;
        }

        imgproc::rectangle( &mut frame,
                             Rect::new (
                                 max_point.x,
                                 max_point.y,
                                 template.cols (),
                                 template.rows ()
                             ),
                             Scalar::all(0.0),
                             2,
                             8,
                             0 )?;

        highgui::imshow(window_name, &frame)?;
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}
