use {
    opencv::{
        core::{self, Scalar, Point, Rect},
        objdetect,
        imgproc,
        types,
        prelude::*,
        videoio,
        highgui
    },
    timm_barth
    // utils
};

// TODO : coll of rects
fn move_eye_center (point: &Point, face: &Rect, eye: &Rect)
                    -> opencv::Result<Point> {
    Ok (Point::new (face.tl ().x
                    + eye.tl ().x
                    + point.x as i32,
                    face.tl ().y
                    + eye.tl ().y
                    + point.y as i32))
}

/*
 * Track pupils movements
 * Move cursor
 */
pub fn run () -> opencv::Result<()> {
    let face_detector_name : &str = "/opt/opencv/opencv-4.2.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    let eyes_detector_name : &str = "/opt/opencv/opencv-4.2.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    let camera_window_name = "camera";

    highgui::named_window(camera_window_name, highgui::WINDOW_AUTOSIZE)?;

    let (face_features, eyes_features, mut cam) = {
        (
            core::find_file(face_detector_name, true, false)?,
            core::find_file(eyes_detector_name, true, false)?,
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?,  // 0 is the default camera
        )
    };

    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    let mut face_detector_model : objdetect::CascadeClassifier = objdetect::CascadeClassifier::new(&face_features)?;
    let mut eyes_detector_model : objdetect::CascadeClassifier = objdetect::CascadeClassifier::new(&eyes_features)?;

    loop {

        let mut frame = Mat::default()?;
        cam.read(&mut frame)?;

        let enhanced_frame = enhance_frame (&frame)?;
        let faces = detect_faces (&enhanced_frame,
                                  &mut face_detector_model)?;

        if faces.len () > 0 {
            // region of interest (submatrix), first detected face
            let face_region = Mat::roi (&enhanced_frame, faces.get (0)?)?;
            let face = faces.get (0)?;

            imgproc::rectangle(
                &mut frame,
                Rect::new (
                    face.tl ().x,
                    face.tl ().y,
                    face.width,
                    face.height
                ), // eye
                core::Scalar::new(0f64, -1f64, -1f64, -1f64),
                1, // thickness
                8, // line type
                0 // shift
            )?;

            // detect eyes
            let eyes = detect_eyes (&face_region,
                                    &mut eyes_detector_model)?;

            if eyes.len () == 2 {
                // let face = faces.get (0)?;

                // draw eyes
                for eye in eyes.iter () {
                    imgproc::rectangle(
                        &mut frame,
                        Rect::new (
                            face.tl ().x + eye.tl ().x,
                            face.tl ().y + eye.tl ().y,
                            eye.width,
                            eye.height
                        ), // eye
                        core::Scalar::new(0f64, -1f64, -1f64, -1f64),
                        1, // thickness
                        8, // line type
                        0 // shift
                    )?;
                }

                let left_eye = eyes.get (0)?;
                let left_eye_region = Mat::roi (&face_region, left_eye)?;
                let left_eye_center = timm_barth::find_eye_center (&left_eye_region, left_eye.width)?;
                let left_eye_center_moved = move_eye_center (&left_eye_center, &face, &left_eye)?;

                let right_eye = eyes.get (1)?;
                let right_eye_region = Mat::roi (&face_region, right_eye)?;
                let right_eye_center = timm_barth::find_eye_center (&right_eye_region, right_eye.width)?;
                let right_eye_center_moved = move_eye_center (&right_eye_center, &face, &right_eye)?;

                imgproc::circle(
                    &mut frame,
                    left_eye_center_moved,
                    1,
                    Scalar::new(0f64, 0f64, 255f64, 0f64),
                    1,
                    8,
                    0)?;

                imgproc::circle(
                    &mut frame,
                    right_eye_center_moved,
                    1,
                    Scalar::new(0f64, 0f64, 255f64, 0f64),
                    1,
                    8,
                    0)?;

            }
        }

        highgui::imshow(camera_window_name, &frame)?;
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}

fn enhance_frame (frame : &Mat)
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

fn detect_faces (frame : &Mat,
                 face_model : &mut objdetect::CascadeClassifier)
                 -> opencv::Result<types::VectorOfRect> {
    let mut faces = types::VectorOfRect::new();

    face_model.detect_multi_scale(
        &frame, // input image
        &mut faces, // output : vector of rects
        1.1, // scaleFactor: The classifier will try to upscale and downscale the image by this factor
        2, // minNumNeighbors: How many true-positive neighbor rectangles do you want to assure before predicting a region as a face? The higher this face, the lower the chance of detecting a non-face as face, but also lower the chance of detecting a face as face.
        objdetect::CASCADE_SCALE_IMAGE,
        core::Size {
            width: 150,
            height: 150
        }, // min_size. Objects smaller than that are ignored (poor quality webcam is 640 x 480, so that should do it)
        core::Size {
            width: 0,
            height: 0
        } // max_size
    )?;

    Ok (faces)
}

fn detect_eyes (frame : &Mat,
                eyes_model : &mut objdetect::CascadeClassifier)
                -> opencv::Result<types::VectorOfRect> {

    // let mut frame_smoothed = Mat::new_rows_cols_with_default(frame.rows (),
    //                                                      frame.cols (),
    //                                                      frame.typ ()?,
    //                                                      Scalar::all(.0))?;

    // let sigma = K_SMOOTH_FACE_FACTOR * frame.width;
    // imgproc::gaussian_blur(
    //     &frame, //src: &dyn ToInputArray,
    //     &mut frame_smoothed, //dst: &mut dyn ToOutputArray,
    //     core::Size {
    //         width: 0,
    //         height: 0
    //     }, //ksize: Size,
    //     sigma, //: f64,
    //     sigma,// f64,
    //     border_type: i32
    // );

    let mut eyes = types::VectorOfRect::new();

    eyes_model.detect_multi_scale(
        &frame,
        &mut eyes,
        1.1,
        2,
        objdetect::CASCADE_SCALE_IMAGE,
        core::Size {
            width: 30,
            height: 30
        }, // min_size
        core::Size {
            width: 0,
            height: 0
        }
    )?;

    Ok(eyes)
}

// fn detect_iris (frame : &Mat)
//                 -> opencv::Result<types::VectorOfPoint3f> {

//     // collection of (x,y, radius)
//     let mut circles = types::VectorOfPoint3f::new();

//     imgproc::hough_circles(
//         &frame,
//         &mut circles,
//         imgproc::HOUGH_GRADIENT, // method
//         1.0, // dp, inverse ratio of the accumulator resolution
//         frame.cols () as f64 / 8.0, // min_dist between the center of one circle and another
//         250.0, //  threshold of the edge detector
//         5.0, // min_area of a circle in the image
//         // 0,
//         // 0
//         frame.rows () / 16, // min_radius of a circle in the image
//         frame.rows () / 4 // max_radius of a circle in the image
//     )?;

//     Ok (circles)
// }
