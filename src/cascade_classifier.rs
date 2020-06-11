use {
    opencv::{
        core,
        objdetect,
        imgproc,
        types,
        prelude::*,
        videoio,
        highgui
    }
};

/*
 * Demonstrates using the CascadeClassifier class to detect objects (Face + eyes) in a video stream.
 * https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
 * https://docs.rs/opencv/0.33.0/opencv/objdetect/struct.CascadeClassifier.html
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

        let mut gray = Mat::default()?;
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0
        )?;

        let mut reduced = Mat::default()?;
        imgproc::resize(
            &gray,
            &mut reduced,
            core::Size {
                width: 0,
                height: 0
            },
            0.25f64,
            0.25f64,
            imgproc::INTER_LINEAR
        )?;

        let mut faces = types::VectorOfRect::new();

        // https://docs.rs/opencv/0.33.0/opencv/objdetect/prelude/trait.CascadeClassifierTrait.html#method.detect_multi_scale
        face.detect_multi_scale(
            &reduced,
            &mut faces,
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

        println!("faces: {}", faces.len());

        for face in faces {
            println!("face {:?}", face);
            let scaled_face = core::Rect {
                x: face.x * 4,
                y: face.y * 4,
                width: face.width * 4,
                height: face.height * 4,
            };
            imgproc::rectangle(
                &mut frame,
                scaled_face,
                core::Scalar::new(0f64, -1f64, -1f64, -1f64),
                1,
                8,
                0
            )?;
        }

        highgui::imshow(window, &frame)?;

        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    Ok(())
}
