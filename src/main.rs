extern crate opencv;
// use opencv::core as cv;
use opencv::{highgui, imgcodecs};

fn main() {

    let image = imgcodecs::imread("lena.jpg", 0).unwrap();
    highgui::named_window("hello opencv!", 0).unwrap();
    highgui::imshow("hello opencv!", &image).unwrap();
    highgui::wait_key(10000).unwrap();

}
