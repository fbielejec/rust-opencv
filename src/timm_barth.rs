use {
    std::collections::{VecDeque},
    opencv:: {
        imgproc,
        highgui,
        prelude::*,
        core::{self, Point, Scalar, CV_64F, Rect}
    }
};

const K_GRADIENT_THRESHOLD: f64 = 50.0;
const K_WEIGHT_BLUR_SIZE: i32 = 5;
const K_ENABLE_WEIGHT: bool = true;
const K_WEIGHT_DIVISOR: f64 = 1.0;
const K_POST_PROCESS_THRESHOLD: f64 = 0.97;
const K_ENABLE_POST_PROCESS: bool = true;
const K_FAST_EYE_WIDTH: i32 = 50;

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

fn unscale_point(p: &Point, width: i32)
                 -> opencv::Result<Point> {
    let ratio: f64 = K_FAST_EYE_WIDTH as f64 / width as f64;
    let x: i32 = (p.x as f64 / ratio) .round () as i32;
    let y: i32 = (p.y as f64 / ratio) .round () as i32;
    Ok (Point::new (x, y))
}

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

            // println!("magnitude: {}", magnitude);

            dx = dx / magnitude;
            dy = dy / magnitude;

            let dot_product: f64 = (dx * gx + dy * gy).max (0.0);

            // println!("dot_product: {}", dot_product);

            // square and multiply by the weight
            if K_ENABLE_WEIGHT {
                out_row[cx as usize] += dot_product * dot_product * (weight_row [cx as usize] as f64 / K_WEIGHT_DIVISOR);
            } else {
                out_row[cx as usize] += dot_product * dot_product;
            }
            // println!("out_row: {}", out_row[cx as usize]);
        }
    }

    Ok (())
}

fn scale_to_fast_size(src: &Mat, mut dst: &mut Mat)
                      -> opencv::Result<()> {
    imgproc::resize(&src,
                    &mut dst,
                    core::Size {
                        width: K_FAST_EYE_WIDTH,
                        height: K_FAST_EYE_WIDTH
                        // height: (K_FAST_EYE_WIDTH / src.cols ()) * src.rows ()
                    },
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR)?;

    Ok (())
}

pub fn find_eye_center (frame : &Mat, frame_width: i32)
                        -> opencv::Result<Point> {
    let mut eye_region = Mat::default ()?;
    scale_to_fast_size (&frame, &mut eye_region)?;

    //-- Find the gradient
    let mut gradient_x : Mat = mat_gradient (&eye_region)?;
    let mut gradient_y = Mat::new_rows_cols_with_default(eye_region.rows (),
                                                         eye_region.cols (),
                                                         f64::typ (),
                                                         Scalar::default ())?;

    core::transpose (&eye_region, &mut gradient_y)?;
    gradient_y = mat_gradient (&gradient_y)?;
    core::transpose (&gradient_y.clone ()?, &mut gradient_y)?;

    // highgui::imshow("DEBUG1", &gradient_x)?;
    // highgui::move_window("DEBUG1", 10, 600);

    // highgui::imshow("DEBUG2", &gradient_y)?;
    // highgui::move_window("DEBUG2", 10, 800);

    // println!("X {:#?}", gradient_x);
    // println!("Y {:#?}", gradient_y);

    let mut magnitudes = gradient_x.clone ()?;
    core::magnitude (&gradient_x, &gradient_y, &mut magnitudes)?;

    // highgui::imshow("DEBUG1", &magnitudes)?;

    let gradient_threshold = compute_dynamic_threshold (&magnitudes, K_GRADIENT_THRESHOLD)?;

    // println!("threshold {}", gradient_threshold);

    // normalize
    for y in 0..eye_region.rows() {
        let x_row = gradient_x.at_row_mut::<f64>(y)?;
        let y_row = gradient_y.at_row_mut::<f64>(y)?;
        let m_row = magnitudes.at_row::<f64>(y)?;

        for x in 0..eye_region.cols() as usize {
            let magnitude = m_row [x];
            if magnitude > gradient_threshold {
                x_row [x] = x_row [x] / magnitude;
                y_row [x] = y_row [x] / magnitude;
            } else {
                x_row [x] = 0.0;
                y_row [x] = 0.0;
            }
        }
    }

    // highgui::imshow("DEBUG1", &gradient_x)?;
    // highgui::move_window("DEBUG1", 10, 600);

    // highgui::imshow("DEBUG2", &gradient_y)?;
    // highgui::move_window("DEBUG2", 10, 800);

    // create a blurred and inverted image for weighting
    let mut weights = Mat::default ()?;

    imgproc::gaussian_blur(
        &eye_region,
        &mut weights,
        core::Size {
            width: K_WEIGHT_BLUR_SIZE,
            height: K_WEIGHT_BLUR_SIZE
        },
        0.0,
        0.0,
        core::BORDER_DEFAULT
    )?;

    // println!("weights {:#?}", weights);

    for y in 0..weights.rows() {
        let ncols = weights.cols();
        let row = weights.at_row_mut::<u8>(y)?;
        for x in 0..ncols as usize {
            // dark centres are more likely than bright centres
            row[x] = 255u8 - row[x];
        }
    }

    // highgui::imshow("DEBUG", &weights)?;

    // run the algorithm for each possible gradient location
    let mut out_sum = Mat::new_rows_cols_with_default(eye_region.rows (),
                                                      eye_region.cols (),
                                                      f64::typ (),
                                                      // start with all values 0
                                                      Scalar::all (0.0))?;

    // NOTE: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.

    // println!("Eye Size: {} {}", out_sum.cols (), out_sum.rows ());

    for y in 0..weights.rows() {
        let x_row = gradient_x.at_row::<f64>(y)?;
        let y_row = gradient_y.at_row::<f64>(y)?;
        for x in 0..weights.cols() as usize {
            let g_x: f64 = x_row[x];
            let g_y: f64 = y_row[x];
            if g_x == 0.0 && g_y == 0.0 {
                continue;
            }
            test_possible_centers_formula(x as i32, y as i32, &weights, g_x, g_y, &mut out_sum)?;
        }
    }

    // highgui::imshow("DEBUG", &out_sum)?;

    // scale all the values down, averaging them
    let num_gradients = weights.rows () * weights.cols () ;
    let mut out = Mat::default ()?;
    out_sum.convert_to (&mut out, CV_64F, 1.0 / num_gradients as f64, 0.0)?;

    // highgui::imshow("DEBUG", &out)?;

    // find the maximum point
    let mut max_point = Point::default ();
    let mut max_value = 0.0;

    core::min_max_loc(
        &out,
        &mut 0.0, // NULL,
        &mut max_value,
        &mut Point::default (), // NULL,
        &mut max_point,
        &core::no_array ()?
    )?;

    // TODO : test
    if K_ENABLE_POST_PROCESS {
        // flood fill the edges
        let mut flood_clone : Mat = Mat::default ()?;
        let flood_threshold : f64 = max_value * K_POST_PROCESS_THRESHOLD;
        imgproc::threshold(&out, &mut flood_clone, flood_threshold, 0.0f64, imgproc::THRESH_TOZERO)?;

        let mask = flood_kill_edges (&mut flood_clone)?;

        highgui::imshow("DEBUG", &out)?;
        // highgui::imshow("DEBUG", &mask)?;

        core::min_max_loc(
            &out,
            &mut 0.0, // NULL,
            &mut max_value,
            &mut Point::default (), // NULL,
            &mut max_point,
            &mask)?;
    }

    Ok (unscale_point (&max_point, frame_width)?)
}

fn is_point_in_mat (p: &Point, mat: &Mat)
                    -> opencv::Result<bool> {
    Ok (p.x >= 0 && p.x < mat.cols () && p.y >= 0 && p.y < mat.rows ())
}

// TODO: test
fn flood_kill_edges (mat: &mut Mat)
                     -> opencv::Result<Mat> {
    let mut mask = Mat::new_rows_cols_with_default(mat.rows (),
                                                   mat.cols (),
                                                   u8::typ () ,
                                                   // start with all values 255
                                                   Scalar::all (255f64))?;

    let mut queue: VecDeque<Point> = VecDeque::new();
    queue.push_back (Point::new (0, 0));

    while !queue.is_empty () {
        let p: Point = queue.pop_front ().unwrap ();
        if mat.at_pt::<f64>(p)? == &0.0f64 {
            continue;
        }

        // add in every direction
        let mut np = Point::new (p.x + 1, p.y); // right
        if is_point_in_mat(&np, &mat)? {
            queue.push_back (np);
        }

        np.x = p.x - 1; np.y = p.y; // left
        if is_point_in_mat(&np, &mat)? {
            queue.push_back (np);
        }

        np.x = p.x; np.y = p.y - 1; // up
        if is_point_in_mat(&np, &mat)? {
            queue.push_back (np);
        }

        // kill it
        *mat.at_pt_mut::<f64> (p)? = 0.0f64;
        *mask.at_pt_mut::<u8> (p)? = 0;
    }

    Ok (mask)
}
