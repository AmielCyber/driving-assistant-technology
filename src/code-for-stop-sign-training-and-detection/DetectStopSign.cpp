#include "DetectStopSign.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

#include <iomanip>
#include <sstream>

DetectStopSign::DetectStopSign(
    const std::string &detector_path,
    const bool has_log,
    const std::string &log_filename)
    : has_log{has_log} {

  if (!hog.load(detector_path)) {
    throw std::runtime_error(
        "Could not load HOG stop-sign detector: " +
        detector_path);
  }

  hog.nlevels = number_of_levels;

  // Do not create or open a CSV file when logging is disabled.
  if (!this->has_log) {
    return;
  }

  log_file.open(
      log_filename,
      std::ios::out | std::ios::trunc);

  if (!log_file.is_open()) {
    throw std::runtime_error(
        "Could not open stop-sign CSV log file: " +
        log_filename);
  }

  // Write the CSV column headings once.
  log_file
      << "frame_number,"
      << "detection_index,"
      << "red_percent,"
      << "white_percent,"
      << "center_x,"
      << "center_y,"
      << "vertex_count,"
      << "octagon_like\n";
}

cv::Mat DetectStopSign::process(cv::Mat &frame) {
  // Return immediately if the supplied frame contains no image data.
  if (frame.empty()) {
    return frame;
  }

  // Advance the internal frame counter only when logging is enabled.
  const std::uint64_t current_frame_number =
      has_log ? frame_number++ : 0;

  // Define the centered vertical region searched by HOG.
  const cv::Rect road_sign_roi_rectangle =
      getMiddleVerticalRegionOfInterest(
          frame.size(),
          roi_height_fraction);

  // Create a cropped Mat view of the selected frame region.
  cv::Mat road_sign_roi_mat =
      frame(road_sign_roi_rectangle);

  // Copy the cropped region into a UMat so the mask and HOG
  // operations use the same representation as the functional version.
  cv::UMat road_sign_roi;
  road_sign_roi_mat.copyTo(road_sign_roi);

  // Create red and white masks from the same ROI searched by HOG.
  const cv::UMat red_mask =
      createRedColorMask(road_sign_roi);

  const cv::UMat white_mask =
      createWhiteColorMask(road_sign_roi);

  // Use the cropped color ROI directly as the HOG input.
  cv::UMat hog_input;
  road_sign_roi.copyTo(hog_input);

  const cv::Size window_stride(
      win_stride_width,
      win_stride_height);

  // Raw HOG candidates are expressed in ROI coordinates.
  std::vector<cv::Rect> hog_candidates;

  hog.detectMultiScale(
      hog_input,
      hog_candidates,
      hit_threshold,
      window_stride,
      cv::Size(0, 0),
      pyramid_scale,
      group_threshold);

  std::vector<StopSignDetection> accepted_detections;

  for (const cv::Rect &candidate : hog_candidates) {
    // Measure the proportion of red pixels inside this HOG box.
    const double red_pixel_ratio =
        calculateMaskPixelRatio(
            red_mask,
            candidate);

    // Measure the proportion of white pixels inside this HOG box.
    const double white_pixel_ratio =
        calculateMaskPixelRatio(
            white_mask,
            candidate);

    int detected_vertex_count = 0;

    // Test whether the largest red contour resembles a stop sign.
    const bool octagon_like =
        isOctagonLikeRedRegion(
            red_mask,
            candidate,
            detected_vertex_count);

    // Reject candidates containing too little red.
    if (red_pixel_ratio < minimum_red_pixel_ratio) {
      continue;
    }

    // Reject candidates containing too little white.
    if (white_pixel_ratio < minimum_white_pixel_ratio) {
      continue;
    }

    // Reject candidates that fail the shape test when enabled.
    if (require_octagon_shape && !octagon_like) {
      continue;
    }

    // Convert the accepted rectangle from ROI coordinates
    // back into full-frame coordinates.
    const cv::Rect detection_in_full_frame(
        candidate.x + road_sign_roi_rectangle.x,
        candidate.y + road_sign_roi_rectangle.y,
        candidate.width,
        candidate.height);

    accepted_detections.push_back(
        StopSignDetection{
            detection_in_full_frame,
            red_pixel_ratio,
            white_pixel_ratio,
            detected_vertex_count,
            octagon_like});
  }

  // Write one CSV row for every accepted stop-sign detection.
    if (has_log) {
    log_detections(
        current_frame_number,
        accepted_detections);
    }

  // Draw only candidates that passed every enabled verification stage.
  draw_detections(
      frame,
      accepted_detections);

  return frame;
}

std::string DetectStopSign::get_feature_name() {
  return name;
}

void DetectStopSign::log_detections(
    const std::uint64_t current_frame_number,
    const std::vector<StopSignDetection> &detections) {

  // Logging was not requested, or the file was not opened.
  if (!has_log || !log_file.is_open()) {
    return;
  }

  for (std::size_t index = 0;
       index < detections.size();
       ++index) {

    const StopSignDetection &detection =
        detections[index];

    // Report the center of the bounding box as the
    // approximate detection location.
    const int center_x =
        detection.rectangle.x +
        detection.rectangle.width / 2;

    const int center_y =
        detection.rectangle.y +
        detection.rectangle.height / 2;

    // Convert ratios such as 0.1842 into percentages
    // such as 18.42.
    const double red_percent =
        detection.red_pixel_ratio * 100.0;

    const double white_percent =
        detection.white_pixel_ratio * 100.0;

    log_file
        << current_frame_number << ','
        << index + 1 << ','
        << std::fixed
        << std::setprecision(2)
        << red_percent << ','
        << white_percent << ','
        << center_x << ','
        << center_y << ','
        << detection.vertex_count << ','
        << std::boolalpha
        << detection.octagon_like
        << '\n';
  }
}

void DetectStopSign::draw_detections(
    cv::Mat &frame,
    const std::vector<StopSignDetection> &detections) {

  const cv::Scalar detection_color(0, 255, 0);

  for (const StopSignDetection &detection : detections) {
    // Draw the accepted stop-sign bounding rectangle.
    cv::rectangle(
        frame,
        detection.rectangle,
        detection_color,
        3);

    // Keep the STOP SIGN label inside the image.
    const int label_y =
        std::max(
            detection.rectangle.y - 5,
            20);

    cv::putText(
        frame,
        "STOP SIGN",
        cv::Point(
            detection.rectangle.x,
            label_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        detection_color,
        2,
        cv::LINE_AA);

    // Build the color and contour diagnostic text.
    std::stringstream diagnostic_text;

    diagnostic_text
        << std::fixed
        << std::setprecision(2)
        << "R:" << detection.red_pixel_ratio
        << " W:" << detection.white_pixel_ratio
        << " V:" << detection.vertex_count
        << " O:"
        << (detection.octagon_like ? "Y" : "N");

    // Keep the diagnostic text inside the bottom of the frame.
    const int diagnostic_y =
        std::min(
            detection.rectangle.y +
                detection.rectangle.height + 20,
            frame.rows - 5);

    cv::putText(
        frame,
        diagnostic_text.str(),
        cv::Point(
            detection.rectangle.x,
            diagnostic_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        detection_color,
        1,
        cv::LINE_AA);
  }
}

// Returns a rectangular Region of Interest (ROI) covering
// the bottom portion of the input frame.
cv::Rect DetectStopSign::getBottomVerticalRegionOfInterest(
    // Size of the full image or video frame.
    const cv::Size& frameSize,

    // Fraction (0.0-1.0) of the frame height to keep.
    // Example:
    // 0.40 = keep the bottom 40% of the image.
    double retainedHeightFraction
)
{
    // Verify that the requested fraction is valid.
    // It must be greater than 0 and no larger than 1.
    if (retainedHeightFraction <= 0.0 ||
        retainedHeightFraction > 1.0)
    {
        // Stop execution and report an error if the
        // fraction is outside the valid range.
        throw std::invalid_argument(
            "ROI retained height fraction must be greater than 0 and at most 1."
        );
    }

    // Compute the ROI height in pixels.
    //
    // Example:
    // Frame height = 1080 pixels
    // Fraction = 0.40
    // ROI height = 1080 × 0.40 = 432 pixels
    const int roiHeight = static_cast<int>(
        frameSize.height * retainedHeightFraction
    );

    // Start the ROI so that it ends exactly at the bottom of the frame.
    // Compute the y-coordinate where the ROI begins. Open CV Y-Pixel indices start at 0
    // Remember we are extracting the bottom portion so it will go down 
    // from the start we extract, called roiY, to the bottom of the image, which is frame height - 1
    //
    // Since we want the ROI to end at the bottom of the image,
    // subtract the ROI height from the full image height.
    //
    // Example:
    // Frame height = 1080
    // ROI, region of interest, height = 432
    // region of interest starting y-value = total frame size height - height of the specified ROI
    // ROI starts at y = 648
    const int roiY = frameSize.height - roiHeight;

    // Return the rectangular ROI using the Rect constructor
    // to create a new Rect object. The parameters are:
    //
    // Rect(x, y, width, height)
    //
    // x = 0
    //     Begin at the left edge of the frame.
    //
    // y = roiY
    //     Begin at the calculated vertical position.
    //
    // width = frameSize.width
    //     Keep the full image width.
    //
    // height = roiHeight
    //     Keep only the selected bottom portion of the frame.
    // parameters for Rect constructor are (x, y, width, height)
    // or equivalently (left edge, top edge, how wide, how tall)
    return cv::Rect(
        0,                  // Begin at left edge
        roiY,               // for ex. Start at 60% when retaining bottom 40%
        frameSize.width,    // Keep the full width
        roiHeight           // Only keep the selected portion of height
    );
}

// Returns a Region of Interest (ROI) centered vertically within the frame.
// therefore, we have an used portion equally split in the top and bottom of the image.
// The ROI keeps the full image width while retaining only a specified
// percentage of the image height.
cv::Rect DetectStopSign::getMiddleVerticalRegionOfInterest(
    // Size (width and height) of the full image or video frame.
    const cv::Size& frameSize,

    // Fraction (0.0-1.0) of the frame height to retain.
    //
    // Example:
    // 0.30 = keep the middle 30% of the image height.
    double retainedHeightFraction
)
{
    // Verify that the requested ROI height fraction is valid.
    // It must be greater than 0 and no greater than 1.
    if (retainedHeightFraction <= 0.0 ||
        retainedHeightFraction > 1.0)
    {
        // Stop execution and report an error if an invalid
        // fraction was provided.
        throw std::invalid_argument(
            "ROI retained height fraction must be greater than 0 and at most 1."
        );
    }

    // Calculate the total ROI height in pixels.
    //
    // Example:
    // Frame height = 1080 pixels
    // ROI fraction = 0.30
    // ROI height = 1080 × 0.30 = 324 pixels
    const int roiHeight = static_cast<int>(
        frameSize.height * retainedHeightFraction
    );

    // Center the ROI vertically.
    // The unused space is divided equally between the top and bottom.

    // Compute the y-coordinate where the centered ROI begins. (AKA the TOP EDGE)
    //
    // OpenCV pixel indices start at y = 0 at the top of the image
    // and increase moving downward.
    //
    // To center the ROI vertically, subtract the ROI height from the
    // total frame height to determine the unused space, then divide
    // that unused space equally between the top and bottom.
    //
    // Example:
    // Frame height = 1080 pixels
    // ROI height = 324 pixels
    // Unused space = 1080 - 324 = 756 pixels
    // Top unused space = 756 / 2 = 378 pixels
    // Therefore, the ROI begins at y = 378, called variable roiY, which is the top edge of the ROI
    // and ends at y = 378 + 324 = 702, which is the bottom edge of the ROI.
    const int roiY = (
        frameSize.height - roiHeight
    ) / 2;

    // Return the rectangular Region of Interest.
    //
    // Rect(x, y, width, height)
    // or equivalently (left edge, top edge, how wide, how tall)
    //
    // x = 0
    //     Begin at the left edge of the image.
    //
    // y = roiY
    //     Begin at the calculated centered vertical position.
    //
    // width = frameSize.width
    //     Keep the entire image width.
    //
    // height = roiHeight
    //     Keep only the selected percentage of the image height.
    return cv::Rect(
        0,                  // Keep the full width from the left edge
        roiY,               // Start after the discarded top section, starting y value of ROI
        frameSize.width,    // Keep 100% of the width
        roiHeight           // Keep the selected percentage of height, using the length of roiHeight
    );
}

// Creates and returns a binary mask containing red pixels from
// the input RGB image.
//
// White pixels (255) represent pixels classified as red.
// Black pixels (0) represent pixels that are not classified as red.
cv::UMat DetectStopSign::createRedColorMask(const cv::UMat& bgrImage) // bgrImage is the input image stored in OpenCV's default BGR color space.
{
    // Create an empty image that will store the HSV version
    // of the input image.
    // HSV works better than RGB for color segmentation because it separates color (hue) from intensity (value).
    cv::UMat hsvImage;

    // Convert the input image from RGB to HSV.
    //
    // HSV separates color (Hue) from brightness (Value),
    // making color thresholding much more reliable than
    // using the original BGR color space.
    cv::cvtColor(
        bgrImage,
        hsvImage,
        cv::COLOR_BGR2HSV
    );

    // Create three binary images (masks).
    //
    // lowerRedMask:
    //     Stores pixels detected in the lower red HSV range.
    //
    // upperRedMask:
    //     Stores pixels detected in the upper red HSV range.
    //
    // combinedRedMask:
    //     Stores the final red mask after combining
    //     both red hue ranges.
    //
    // for example, red is stored in the 0-180 degrees range and it
    // repeats on both sides of the hue spectrum, so we need to combine both ranges to get all red pixels.
    cv::UMat lowerRedMask; // lower red hue on the left end of the spectrun
    cv::UMat upperRedMask; // upper red hue on the right end of the spectrum
    // we must iterate lineraly through the hue spectrum to get all red pixels, 
    //so we combine both masks into one final mask
    cv::UMat combinedRedMask;

    // Red appears at both ends of OpenCV's hue range.

    // Red is unique because it wraps around the HSV color wheel.
    // Therefore, OpenCV represents red using two separate hue ranges.
    //
    // This first range captures reds near hue = 0.
    //
    // H = Hue        ---  What color is it?
    // S = Saturation ---  How pure or vivid is the color?
    // V = Value      ---  How bright is the color?
    //
    // sample hues on the color wheel
    // 0 - 180 degrees to save money NOT 0 - 360 degrees
    //
    // OpenCV Hue
    //
    //  0       Red
    //  15      Orange
    //  30      Yellow
    //  60      Green
    //  90      Cyan
    //  120     Blue
    //  150     Purple
    //  180     Red again
    //
    // hence
    //
    // 0 ---------------------- 180
    // ^                         ^
    //Red                       Red
    //
    // HSV Lower Bound:
    // H = 0  --- HUE -- the starting hue for red on the color spectrum scale
    // S = 60 --- SATURATION
    // V = 40 --- VALUE
    //
    // HSV Upper Bound:
    // H = 12  --- HUE -- the ending hue for red on the color spectrum scale
    // S = 255 --- SATURATION
    // V = 255 --- VALUE
    //
    // ex. Scalar(hue, saturation, value)
    //
    // Every pixel inside this HSV range becomes white (255).
    // Every other pixel becomes black (0).
    //
    // function signature for inRange is:
    //
    // inRange(
    //     inputImage,
    //     lowerBound,
    //     upperBound,
    //     outputMask
    // );
    // where a mask is created of 0's and 255's based on the input image BY
    // comparing every pixel in the input image against the lower and upper bounds.
    // and the input image is left unchanged
    cv::inRange(
        hsvImage,
        cv::Scalar(0, 60, 40),
        cv::Scalar(12, 255, 255),
        lowerRedMask
    );

    // Similarly
    //
    // Detect the second red hue range near hue = 180.
    //
    // Since OpenCV's hue values wrap around,
    // red also exists near the upper end of the hue scale.
    //
    // Every pixel inside this range becomes white (255).
    // Every other pixel becomes black (0).
    //
    // Note that whites and blacks are defaults to the 
    // inRange function when creating a mask
    cv::inRange(
        hsvImage,
        cv::Scalar(168, 60, 40),
        cv::Scalar(180, 255, 255),
        upperRedMask
    );

    // Combine both red masks into one binary image.
    //
    // A pixel will be white if it belongs to either
    // the lower-red mask OR the upper-red mask.
    //
    // and black if it does not belong to the lower red mask 
    // AND black if it does NOT belong to the upper red mask
    cv::bitwise_or(
        lowerRedMask,
        upperRedMask,
        combinedRedMask
    );

    // Connect nearby red pixels belonging to the same sign.
    //
    // getStructuringElement function signature
    //
    // uses 
    //
    // getStructuringElement( shape, kernalSize)
    //
    // our kernal size is 3 x 3
    // our shape is MORPH_ELLIPSE
    cv::Mat closingKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );

    // Perform CLOSING on the combined red mask
    //
    // Closing consists of:
    // 1. Dilation (make bigger, nearby pixels are connected)
    // 2. Erosion (make smaller, remove small holes)
    //
    // This helps:
    // • connect nearby red pixels
    // • fill small holes
    // • reduce small gaps caused by shadows,
    //   compression artifacts, or noise
    //
    // This produces a cleaner red mask for
    // later contour extraction.
    //
    // function signature for morphologyEx is:
    //
    // morphologyEx(
    //      inputImage,
    //      outputImage,
    //      operation,
    //      kernel
    // );
    cv::morphologyEx(
        combinedRedMask,
        combinedRedMask,
        cv::MORPH_CLOSE,
        closingKernel
    );

    // Return the completed binary red mask.
    return combinedRedMask;
}

// Creates and returns a binary mask containing white pixels from
// the input BGR image.
//
// White pixels (255) represent pixels classified as white.
// Black pixels (0) represent pixels that are not classified as white.
//
// --- REMEMBER ---
//
// Red is detected primarily by its Hue.
// White is not detected by Hue. 
// Instead, white is detected because it has very little color (low saturation) 
// and is fairly bright (high value)
cv::UMat DetectStopSign::createWhiteColorMask(const cv::UMat& bgrImage) // rgb called brgImage 
//image stored in OpenCV's default BGR color space.
{
    // Create an empty image that will store the HSV version
    // of the input image.
    cv::UMat hsvImage;

    // Convert the input image from BGR to HSV.
    //
    // HSV separates color (Hue) from brightness (Value),
    // making color thresholding much more reliable than
    // using the original BGR color space.
    //
    // function signature for cvtColor is:
    // cvtColor(inputImage, outputImage, conversionCode)
    cv::cvtColor(
        bgrImage,
        hsvImage,
        cv::COLOR_BGR2HSV
    );

    // Create the binary mask that will store pixels
    // classified as white.
    cv::UMat whiteMask;

    /*
     * White has:
     * - low saturation
     * - relatively high brightness/value
     *
     * These values are intentionally permissive at first.
     */

    // Function signature for OpenCV's inRange().
    //
    // inRange(
    //     inputImage,
    //     lowerBound,
    //     upperBound,
    //     outputMask
    // );
    //
    // OpenCV examines every pixel in the input image.
    //
    // If a pixel falls within the specified lower and upper bounds,
    // the corresponding output mask pixel is assigned a value of
    // 255 (white).
    //
    // Otherwise, the corresponding output mask pixel is assigned
    // a value of 0 (black).
    //
    // The input image is NOT modified. Instead, a new binary mask
    // (whiteMask) is created.
    //
    // Unlike red, white does NOT have a unique Hue.
    //
    // White is characterized primarily by:
    // • very low Saturation (little or no color)
    // • relatively high Value (brightness)
    //
    // HSV Lower Bound:
    //
    // H = 0
    // Hue represents the actual color.
    // We begin at 0 because Hue is not important for detecting white.
    //
    // S = 0
    // Saturation measures how colorful a pixel is.
    // White has almost no color, so we allow saturation to start at 0.
    //
    // V = 140
    // Value represents brightness.
    // Require the pixel to be reasonably bright to avoid
    // detecting dark gray or black regions.
    //
    // HSV Upper Bound:
    //
    // H = 180
    // Allow every possible Hue because white has no meaningful Hue.
    //
    // S = 90
    // Allow only relatively low saturation values.
    // Highly saturated pixels are colorful rather than white.
    //
    // V = 255
    // Allow the brightest possible pixels.
    //
    // Note allow every possible Hue for White
    // 0 - 180 degrees on the color wheel
    cv::inRange(
        hsvImage,
        cv::Scalar(0, 0, 140),
        cv::Scalar(180, 90, 255),
        whiteMask
    );

    // Function signature:
    //
    // getStructuringElement(
    //     kernelShape,
    //     kernelSize
    // );
    //
    // Create a small 3×3 elliptical structuring element (kernel).
    //
    // A kernel defines the neighboring pixels considered during
    // morphological image processing.
    cv::Mat closingKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );

    // Function signature:
    //
    // morphologyEx(
    //     inputImage,
    //     outputImage,
    //     morphologicalOperation,
    //     structuringKernel
    // );
    //
    // Apply a morphological closing operation to the white mask.
    //
    // Morphological processing modifies the shape and connectivity
    // of white regions in a binary image.
    //
    // Closing consists of:
    // 1. Dilation
    // 2. Erosion
    //
    // This helps:
    // • connect nearby white pixels
    // • fill small holes
    // • reduce small gaps caused by shadows,
    //   reflections, compression artifacts, or noise
    //
    // A cleaner white mask improves later verification of the
    // STOP letters and white border of the stop sign.
    //
    // The same image is used as both the input and output,
    // so the white mask is modified in place.
    cv::morphologyEx(
        whiteMask,
        whiteMask,
        cv::MORPH_CLOSE,
        closingKernel
    );

    // Return the completed binary white mask.
    return whiteMask;
}

// Calculates the proportion of nonzero pixels inside a specified
// detection rectangle.
//
// This function receives a binary mask in which:
//
// 255 (white) = the pixel passed the color-threshold test
// regardless of which color was tested (red or white).
// 0 (black) = the pixel did not pass the color-threshold test
//
// The function returns a decimal ratio between 0.0 and 1.0.
//
// Example:
// 0.05 means 5% of the pixels inside the detection rectangle
// are white/nonzero in the supplied binary mask
double DetectStopSign::calculateMaskPixelRatio(
    // Binary mask that will be examined.
    const cv::UMat& redOrWhiteMask,

    // Rectangle representing the HOG candidate region that should // be examined inside the binary mask.
    const cv::Rect& detectionRectangle
)
{
    // Restrict the rectangle to valid mask coordinates.
    //
    // Create a rectangle representing the complete valid area
    // of the binary mask.
    //
    // Rect(x, y, width, height)
    //
    // or equivalently Rect(left edge, top edge, how wide, how tall)
    //
    // x = 0
    // Begin at the left edge of the mask.
    //
    // y = 0
    // Begin at the top edge of the mask.
    //
    // width = redMask.cols
    // Include every column in the mask.
    //
    // height = redMask.rows
    // Include every row in the mask.
    //
    // This rectangle therefore covers the entire valid mask area.
    const cv::Rect validImageRectangle(
        0,
        0,
        redOrWhiteMask.cols,
        redOrWhiteMask.rows
    );

    // Find the intersection between the HOG detection rectangle 
    // and the valid mask rectangle. 
    // 
    // For OpenCV Rect objects, the & operator calculates the 
    // overlapping region between two rectangles. 
    // 
    // This protects the program if a HOG detection rectangle extends 
    // partially outside the binary mask. 
    // 
    // Example: 
    // 
    // Detection rectangle: 
    // x = 620, width = 100 
    // 
    // Mask width: 
    // 640 pixels 
    // 
    // The detection would extend to x = 720, which is outside 
    // the mask. The intersection clips it so that only the valid 
    // portion between x = 620 and x = 639 is examined.
    const cv::Rect clippedRectangle =
        detectionRectangle & validImageRectangle;

    // Verify that the clipped rectangle contains a valid area. 
    // 
    // clippedRectangle.empty() is true when the rectangle has 
    // no usable width or height. 
    // 
    // clippedRectangle.area() returns: 
    // 
    // width × height 
    // 
    // If the rectangle is empty or its area is zero or negative, 
    // there are no pixels available to examine.
    if (clippedRectangle.empty() ||
        clippedRectangle.area() <= 0)
    {
        // prevent dividing by zero since if we are
        // here in the code this implies the percentage of pixels
        // to examine is 0% or none
        return 0.0;
    }

    // with our final HOG candidate
    // here is within the ROI
    // and is cropped to match the valid mask area
    //
    // steps are 
    // 1. ROI
    // 2. HOG candidate
    // 3. valid mask area, first red then white
    // 4. clipped rectangle, which is the intersection of the HOG candidate and the valid mask area
    // 5. count the number of non-zero pixels in the clipped rectangle
    // 6. divide the number of non-zero pixels by the total number of pixels in the clipped rectangle to get the ratio
    // 7. return the ratio as a double
    //
    // Extract the portion of the binary mask located inside // the valid clipped detection rectangle.
    // intersection of HOG and valid mask area
    cv::UMat detectionMask =
        redOrWhiteMask(clippedRectangle);

    // Count how many pixels inside the detection mask are nonzero. 
    // 
    // In this binary mask: 
    // 
    // 255 = pixel passed the threshold test - WHITE
    // 0 = pixel did not pass the threshold test - BLACK
    // 
    // Therefore, countNonZero() counts the number of pixels that 
    // were classified as red, white respectively when redOrWhiteMask is supplied. 
    const int redOrWhitePixelCount =
        cv::countNonZero(detectionMask);

    // Formula: 
    // 
    // number of nonzero mask pixels 
    // -------------------------------- 
    // total number of pixels in the rectangle
    //
    // without static_cast<double> we would be doing integer division
    //
    // leading to an incorrect result
    return static_cast<double>(redOrWhitePixelCount) /
           static_cast<double>(clippedRectangle.area());
}

// Determine whether the largest red region inside a HOG detection
// resembles the shape of a stop sign.
//
// That is, whether it is approximately an octagon
// Or equivalently, to be generous and maximize recall, 
//
// this uses contours to evaluate whehter or not the 
// HOG red region has between 6-10 vertices
bool DetectStopSign::isOctagonLikeRedRegion(
    // Binary red mask created earlier.
    const cv::UMat& redMask,

    // Current HOG detection rectangle.
    const cv::Rect& detectionRectangle,

    // Output parameter that stores the detected number of polygon vertices.
    int& detectedVertexCount
)
{
    // initialize the counter to the total detected vertex count to 0
    detectedVertexCount = 0;

    // Limit the detection rectangle to valid mask coordinates.
    const cv::Rect validMaskRectangle(
        0,
        0,
        redMask.cols,
        redMask.rows
    );

    // again perform the intersection of the HOG detection rectangle
    // and the valid mask rectangle
    // ensures we examine only red pixels within
    // the HOG detection rectange
    const cv::Rect clippedRectangle =
        detectionRectangle & validMaskRectangle;

    // Reject invalid or extremely small detection regions.
    if (clippedRectangle.empty() ||
        clippedRectangle.width < 10 ||
        clippedRectangle.height < 10)
    {
        return false;
    }

    /*
     * findContours operates most predictably on a CPU Mat.
     * copyTo also ensures findContours may safely modify its input.
     */

    // Copy the candidate region into a CPU Mat because
    // findContours() operates on Mat objects.
    //
    // and we need the built in findContours() function 
    // of OpenCV to approximate total contour vertices
    cv::Mat candidateMask;
    redMask(clippedRectangle).copyTo(candidateMask);

    // Connect small breaks in the outer red boundary.
    // similar to how getStructuringElement function
    // was used earlier
    cv::Mat closingKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );

    // Close small gaps in the red region before contour detection.
    cv::morphologyEx(
        candidateMask,
        candidateMask,
        cv::MORPH_CLOSE,
        closingKernel
    );

    // Store all detected contours.
    std::vector<std::vector<cv::Point>> contours;

    // Find only the outer contours in the binary mask.
    // which is the intersection of 
    // HOG detection and
    // red mask
    // 
    // converted from Rect object
    //
    // to a Mat object
    cv::findContours(
        candidateMask,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    // Reject if no red contours were found.
    // we do not have an approximate octagon in this case
    if (contours.empty())
    {
        return false;
    }

    // Find the largest red contour inside the HOG box.
    // by iterating through each pixel in the
    // specified region
    // and resetting the largestContourArea and index if we have found one region bigger
    std::size_t largestContourIndex = 0;
    double largestContourArea = 0.0;

    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        const double currentArea =
            cv::contourArea(contours[i]);

        if (currentArea > largestContourArea)
        {
            largestContourArea = currentArea;
            largestContourIndex = i;
        }
    }

    // Compute the total area of the HOG detection rectangle.
    const double detectionArea =
        static_cast<double>(clippedRectangle.area());

    // if we have no hog detection area
    // thus we have no contours
    // and avoid division by zero
    if (detectionArea <= 0.0)
    {
        return false;
    }

    // Compute how much of the HOG box is filled by
    // the largest red contour.
    const double contourFillRatio =
        largestContourArea / detectionArea;

    /*
     * Reject tiny red fragments.
     * This remains equal to your minimum red-ratio starting point.
     */
    //
    // Reject tiny red regions that are unlikely
    // to represent a stop sign.
    if (contourFillRatio < 0.05)
    {
        return false;
    }

    // Retrieve the largest contour
    // from the intersection between HOG region 
    // and red region
    const std::vector<cv::Point>& largestContour =
        contours[largestContourIndex];

    /*
     * Build the convex hull to reduce small dents caused by:
     * - white STOP letters
     * - compression
     * - shadows
     * - fragmented red pixels
     */
    std::vector<cv::Point> contourHull;

    cv::convexHull(
        largestContour,
        contourHull
    );

    // A polygon requires at least three points.
    if (contourHull.size() < 3)
    {
        return false;
    }

    // Compute the contour perimeter.
    const double perimeter =
        cv::arcLength(contourHull, true);

    // Reject invalid contours.
    if (perimeter <= 0.0)
    {
        return false;
    }

    // Store the simplified polygon.
    std::vector<cv::Point> approximatedPolygon;

    /*
     * Epsilon controls polygon simplification.
     *
     * Larger epsilon = fewer vertices.
     * Smaller epsilon = more vertices.
     *
     * Begin with 3% of the contour perimeter.
     */
    cv::approxPolyDP(
        contourHull,
        approximatedPolygon,
        0.03 * perimeter,
        true
    );

    // Store the number of detected polygon vertices.
    detectedVertexCount =
        static_cast<int>(approximatedPolygon.size());

    // Reject invalid polygons.
    // < 3 is NOT a polygon
    if (detectedVertexCount < 3)
    {
        return false;
    }

    // Stop signs should be convex.
    // so if it's not convex, we should reject it
    if (!cv::isContourConvex(approximatedPolygon))
    {
        return false;
    }

    // Compute the bounding rectangle of the polygon.
    const cv::Rect polygonBounds =
        cv::boundingRect(approximatedPolygon);

    // Reject invalid rectangles.
    if (polygonBounds.width <= 0 ||
        polygonBounds.height <= 0)
    {
        return false;
    }

    // Compute the width-to-height ratio.
    const double aspectRatio =
        static_cast<double>(polygonBounds.width) /
        static_cast<double>(polygonBounds.height);

    /*
     * Stop signs are approximately square, but perspective can
     * make them appear somewhat wider or narrower.
     */
    // aspectRatio = 1 is a perfect square
    if (aspectRatio < 0.60 ||
        aspectRatio > 1.50)
    {
        return false;
    }

    /*
     * Do not initially require exactly eight vertices.
     * Real signs may produce 6–10 due to blur and perspective.
     */
    return detectedVertexCount >= 6 &&
           detectedVertexCount <= 10;
}