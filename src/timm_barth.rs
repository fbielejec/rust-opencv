use {
    // std::cmp,
    opencv:: {
        imgproc,
        highgui,
        prelude::*,
        core::{self, Point, Scalar, CV_64F}
    }
};

const K_GRADIENT_THRESHOLD: f64 = 50.0;
const K_WEIGHT_BLUR_SIZE: i32 = 5;
const K_ENABLE_WEIGHT: bool = true;
const K_WEIGHT_DIVISOR: f64 = 1.0;

fn mat_gradient(mat: &Mat)
                -> opencv::Result<Mat> {
    let mut out = Mat::new_rows_cols_with_default(mat.rows (),
                                                  mat.cols (),
                                                  f64::typ (),
                                                  Scalar::default ())?;

    for y in 0..mat.rows() {
        let out_row = out.at_row_mut::<f64>(y)?;
        let mat_row = mat.at_row::<u8>(y)?;

        out_row [0] = mat_row [1] as f64 - mat_row [0] as f64;
        for x in 1..mat.cols() - 1  {
            out_row[x as usize] = (mat_row [(x + 1) as usize] as f64 - mat_row [(x - 1) as usize] as f64) / 2.0;
        }
        out_row[(mat.cols() - 1) as usize ] = mat_row [(mat.cols() - 1) as usize] as f64 - mat_row [(mat.cols() - 2) as usize] as f64;
    }

    Ok (out)
}

fn compute_dynamic_threshold(mat: &Mat, std_dev_factor: f64)
                             -> opencv::Result<f64> {
    let mut mean_magnitude_grad = Mat::default ()?;
    let mut std_dev_magnitude_grad = Mat::default ()?;

    core::mean_std_dev(
        &mat,
        &mut mean_magnitude_grad,
        &mut std_dev_magnitude_grad,
        &core::no_array ()?
    )?;

    let std_dev = std_dev_magnitude_grad.at_2d::<f64> (0,0)? / (mat.cols () as f64 * mat.rows () as f64).sqrt ();
    let mean = mean_magnitude_grad.at_2d::<f64> (0,0)?;

    Ok (std_dev_factor * std_dev + mean)
}

// TODO : bug?
fn test_possible_centers_formula (x: i32, y: i32, weight: &Mat, gx: f64, gy: f64, out: &mut Mat)
                                  -> opencv::Result<()> {
    // for all possible centers
    for cy in 0..out.rows()  {
        let ncol = out.cols();
        let out_row = out.at_row_mut::<f64>(cy)?;
        let weight_row = weight.at_row::<u8>(cy)?;

        for cx in 0..ncol {
            if x == cx && y == cy {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            let mut dx: f64 = x as f64 - cx as f64;
            let mut dy: f64 = y as f64 - cy as f64;
            // normalize d
            let magnitude: f64 = ((dx * dx + dy * dy) as f64).sqrt ();

            dx = dx / magnitude;
            dy = dy / magnitude;

            // let dot_product: f64 = 0.0f64.max (dx * gx + dy * gy);
            let dot_product: f64 =  (dx * gx + dy * gy).max (0.0);

            // square and multiply by the weight
            if K_ENABLE_WEIGHT {
                out_row[cx as usize] += dot_product * dot_product * (weight_row [cx as usize] as f64 / K_WEIGHT_DIVISOR);
            } else {
                out_row[cx as usize] += dot_product * dot_product;
            }
        }
    }

    Ok (())
}

// TODO : implement
pub fn find_eye_center (frame : &Mat)
                        -> opencv::Result<Point> {
    let mut gradient_x : Mat = mat_gradient (&frame)?;
    let mut gradient_y = Mat::new_rows_cols_with_default(frame.rows (),
                                                         frame.cols (),
                                                         f64::typ (),
                                                         Scalar::default ())?;

    core::transpose (&frame, &mut gradient_y)?;
    gradient_y = mat_gradient (&gradient_y)?;
    core::transpose (&gradient_y.clone ()?, &mut gradient_y)?;

    let mut magnitudes = gradient_x.clone ()?;
    core::magnitude (&gradient_x, &gradient_y, &mut magnitudes)?;

    let gradient_threshold = compute_dynamic_threshold (&magnitudes, K_GRADIENT_THRESHOLD)?;

    // normalize
    for y in 0..frame.rows() {
        let x_row = gradient_x.at_row_mut::<f64>(y)?;
        let y_row = gradient_y.at_row_mut::<f64>(y)?;
        let m_row = magnitudes.at_row_mut::<f64>(y)?;

        for x in 0..frame.cols() as usize {
            let magnitude = m_row [x];
            if magnitude < gradient_threshold {
                x_row [x] = x_row [x] / magnitude;
                y_row [x] = y_row [x] / magnitude;
            } else {
                x_row [x] = 0.0;
                y_row [x] = 0.0;
            }
        }
    }

    // highgui::imshow("DEBUG", &gradient_x)?;

    // create a blurred and inverted image for weighting
    let mut weight = Mat::default ()?;
    imgproc::gaussian_blur(
        &frame,
        &mut weight,
        core::Size {
            width: K_WEIGHT_BLUR_SIZE,
            height: K_WEIGHT_BLUR_SIZE
        },
        0.0,
        0.0,
        core::BORDER_DEFAULT
    )?;

    // println!("@@@ DEBUG {:#?}", weight);

    for y in 0..weight.rows() {
        let row = gradient_x.at_row_mut::<f64>(y)?;
        for x in 0..frame.cols() as usize {
            row[x] = 255.0 - row[x];
        }
    }

    // highgui::imshow("DEBUG", &weight)?;

    // run the algorithm
    let mut out_sum = Mat::new_rows_cols_with_default(frame.rows (),
                                                      frame.cols (),
                                                      f64::typ (),
                                                      Scalar::all (0.0))?;
    // Mat::zeros(frame.rows(), frame.cols(), CV_64F)?;

    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.

    // println!("Eye Size: {} {}", out_sum.cols (), out_sum.rows ());

    for y in 0..weight.rows() {
        let x_row = gradient_x.at_row::<f64>(y)?;
        let y_row = gradient_y.at_row::<f64>(y)?;
        for x in 0..weight.cols() as usize {
            let g_x: f64 = x_row[x];
            let g_y: f64 = y_row[x];
            if g_x == 0.0 && g_y == 0.0 {
                continue;
            }
            test_possible_centers_formula(x as i32, y as i32, &weight, g_x, g_y, &mut out_sum)?;
        }
    }

    // scale all the values down, basically averaging them
    let num_gradients: f64 = weight.rows () as f64 * weight.cols () as f64;

    let mut out =
    // Mat::new_rows_cols_with_default(out_sum.rows (),
    //                                           out_sum.cols (),
    //                                           f64::typ (),
    //                                           Scalar::all (0.0))?;
        Mat::default ()?;
    out_sum.convert_to (&mut out, CV_64F, 1.0 / num_gradients, 0.0)?;

    // TODO : sth broken
    highgui::imshow("DEBUG", &out)?;

    // find the maximum point
    // let max_point: Point = Point::default ();
    // let max_value: f64 = 0.0;

    // core::min_max_loc(
    //     &mut out,
    //     // min_val: &mut f64,
    //     // max_val: &mut f64,
    //     // min_loc: &mut Point,
    //     // max_loc: &mut Point,
    //     // mask: &dyn ToInputArray
    // )?;

    // TODO
    Ok (Point::new (1,1))
}
