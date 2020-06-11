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

    let face_detector_name : &str = "/opt/opencv/opencv-4.2.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    let eyes_detector_name : &str = "/opt/opencv/opencv-4.2.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

    let window = "video capture";
    highgui::named_window(window, 1)?;

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

        // TODO : eyes
        let eyes = detect_eyes (&frame,
                                &mut face_detector_model,
                                &mut eyes_detector_model)?;

        if eyes.len () == 2 {

            println!("eyes detected: {:?} {:?}", eyes.get (0)?.x, eyes.get (1)?.x);

            for eye in eyes {
                imgproc::rectangle(
                    &mut frame,
                    eye,
                    core::Scalar::new(0f64, -1f64, -1f64, -1f64),
                    1, // thickness
                    8, // line type
                    0 // shift
                )?;
            }
        }

        highgui::imshow(window, &frame)?;

        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}

fn detect_eyes (frame : &Mat,
                face_model : &mut objdetect::CascadeClassifier,
                eyes_model : &mut objdetect::CascadeClassifier)
                -> opencv::Result<types::VectorOfRect > {

    let mut gray = Mat::default()?;
    imgproc::cvt_color(
        &frame,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0
    )?;

    let mut equalized = Mat::default()?;
    imgproc::equalize_hist (&gray,
                            &mut equalized)?;

    let mut faces = types::VectorOfRect::new();
    face_model.detect_multi_scale(
        &equalized, // input image
        &mut faces, // output : vector of rects
        1.1, // scaleFactor: The classifier will try to upscale and downscale the image by this factor
        2, // minNumNeighbors: How many true-positive neighbor rectangles do you want to assure before predicting a region as a face? The higher this face, the lower the chance of detecting a non-face as face, but also lower the chance of detecting a face as face.
        objdetect::CASCADE_SCALE_IMAGE,
        core::Size {
            width: 150,
            height: 150
        }, // minSize: Minimum possible object size. Objects smaller than that are ignored (poor quality webcam is 640 x 480, so that should do it)
        core::Size {
            width: 0,
            height: 0
        } // maxSize
    )?;

    let mut eyes = types::VectorOfRect::new();

    if faces.len () > 0 {

        // println!("faces detected: {:?}", faces.get (0));

        // region of interest (submatrix)
        let mut face = Mat::roi (&frame, faces.get (0)?)?;

        // let eyes_ref = &mut eyes;

        eyes_model.detect_multi_scale(
            &face,
            &mut eyes,
            // eyes_ref,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size {
                width: 30,
                height: 30
            },
            core::Size {
                width: 0,
                height: 0
            }
        )?;

        // TODO : scale back to frame coords

        // faces[0].tl() + eye.tl(), faces[0].tl() + eye.br()
        // eyes detected:
        // Ok(Rect_ { x: 49, y: 60, width: 44, height: 44 })
        // Ok(Rect_ { x: 122, y: 52, width: 63, height: 63 })

        let mut eyes_scaled = types::VectorOfRect::new();
        for mut eye in eyes.iter () {

        println! ("@@@ {:?}", eye.tl() );

            eye.x = 50;
            eyes_scaled.push (eye);
        }

        // println! ("@@@ {}", eyes_scaled.len () );

        return Ok(eyes_scaled);
    }

    Ok(eyes)
}
