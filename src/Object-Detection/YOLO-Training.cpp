//
//  YOLO-Training.cpp
//  Created by Carlos G Cazares
//  This is a loop program to train YOLO11n with Barkley Deep Drive Data Set
//  No arguments
//  Use Ultralytics directly
//  Program C++ prepare Python Commands end they are running
//  In case the Training fails, the loop resume the training from the last save checkpoint

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Helper to get the timestap and calculate when the training start, how long took and when the training was done
std::string now_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Helper to execute commands to launch Python programs
int run_command(const std::string& cmd) {
    std::cout << "\nRunning:\n" << cmd << "\n\n";
    return std::system(cmd.c_str());
}

// Helper to find if the training check point exists
bool checkpoint_exists(const fs::path& checkpoint) {
    return fs::exists(checkpoint) && fs::is_regular_file(checkpoint);
}

int main() {
#ifdef __APPLE__
    setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1", 1); // use CPU in case MPS failed
#else
    _putenv_s("PYTORCH_ENABLE_MPS_FALLBACK", "1"); // Use MPS by default
#endif

    // This is for the prototype, in the future we can provide these as an arguments.
    // for now this are hard coded to run in a Macbook locally.
    const std::string yolo = "/Users/car/miniforge3/envs/ml/bin/yolo";

    const std::string data_yaml = "bdd_yolo_6cls/data.yaml";
    const std::string project = "runs/detect";
    const std::string run_name = "train_bdd_6cls";

    const fs::path run_dir = fs::path(project) / run_name;
    const fs::path last_pt = run_dir / "weights" / "last.pt";
    const fs::path best_pt = run_dir / "weights" / "best.pt";

    const int max_attempts = 100;

    // Time stamp of the starting point of the training.
    // I may change the functions to use the recommendation for the professor.
    std::string start_time = now_string();
    auto start = std::chrono::steady_clock::now();

    // Displaying the starting time
    std::cout << "Training started at: " << start_time << std::endl;

    // This is our flag control, if we need to resume the training status will tell the program. By defult 1 = Yes resume
    int status = 1;

    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        std::cout << "\nTraining attempt " << attempt << " of " << max_attempts << "\n";

        std::string cmd;

        // This is checking if the training completed before exist best.pt in addition
        // to just check for the last.pt, in case the training failed.
        // these 2 together provide a better logic in case we want to retrain the model
        //if (checkpoint_exists(last_pt) && !checkpoint_exists(best_pt)) {
        if (checkpoint_exists(last_pt)) { // Previous logic was incorrect. Maybe I will remove it
            std::cout << "Checkpoint found. Resuming from:\n"
                      << last_pt.string() << "\n";

            cmd =  // This aprt is to resume the training after a fail.
                "\"" + yolo + "\" "
                "detect train "
                "model=\"" + last_pt.string() + "\" "
                "resume=True";
        } else { // This part is to run the training the first part or to retrain the model
            std::cout << "No checkpoint found. Starting from yolo11n.pt\n";

            cmd = // creating the Python command to launch the training with the hyper-parameters using OS command line approach
                "\"" + yolo + "\" "
                "detect train "
                "model=yolo11n.pt "
                "data=" + data_yaml + " "
                "epochs=300 " // Number of full rounds for the training 100 is not enough. try 300.
                "imgsz=640 " // Image size for analysis
                "batch=8 " // Process 8 images at the same time
                "device=mps " // Use M chips GPU rather than CPU
                "workers=4 " // Using multi-cores to expediate the process
                "cache=False " // prefer  SSD over RAM to avoid
                "amp=False " // Use less complex numbers when is possible, to reduce computing resources
                "close_mosaic=0 "
                "val=False " // Do not run validation during training to save time and run valdiation after training
                "plots=False " // Do not report training stats graphically
                "save_period=1 " // Create a checck point in case the training fails after one epoch
                "name=" + run_name + " "
                "exist_ok=True"; // Overwrite existing project directories instead of automatically creating new, incremented folders
        }

        status = run_command(cmd); // Execute the Python command

        if (status == 0) { // If the training completed normal stop the loop
            break;
        }

        std::cerr << "\nTraining failed with exit code: " << status << "\n";

        if (!checkpoint_exists(last_pt)) { /// If the training failed or was interrupted find the last.pt to restart the training
            std::cerr << "No checkpoint exists yet. Cannot resume.\n";
            return status;
        }

        std::cerr << "Retrying from saved checkpoint...\n";
    }

    if (status != 0) { // To let the maximum number of attempts was reached and training did not complete normal
        std::cerr << "\nTraining failed after max attempts.\n";
        return status;
    }

    // When traiing completes
    std::cout << "\nTraining completed successfully.\n";
    std::cout << "Starting validation automatically...\n";

    // Find the location of the training check point last result (last.pt) or the results of the complete training (best.pt)
    fs::path validation_model = checkpoint_exists(best_pt) ? best_pt : last_pt;

    std::string val_cmd =
        "\"" + yolo + "\" "
        "detect val " // Command to start the validation instead of training.
        "model=\"" + validation_model.string() + "\" " // use the best or last for the valdiation based on above logic.
        "data=" + data_yaml + " " // this is the data structure
        "device=cpu " // Validation is done using CPU
        "imgsz=640"; // Size of valdiation images

    int val_status = run_command(val_cmd);

    if (val_status != 0) { /// Let the user know validation failed.
        std::cerr << "\nValidation failed with exit code: "
                  << val_status << std::endl;
    } else {
        std::cout << "\nValidation completed successfully.\n";
    }

    // Timestamp of when the process completed to calculate elapsed time for the whole process including training and validation
    auto finish = std::chrono::steady_clock::now();
    std::string finish_time = now_string();

    // Compute the elapsed time
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();

    // displayed general stats about the time required for the process as a summary
    std::cout << "Training started at: " << start_time << std::endl;
    std::cout << "Training finished at: " << finish_time << std::endl;
    std::cout << "Elapsed total time: " << elapsed_seconds << " seconds\n";

    std::ofstream log("training_time_log.txt", std::ios::app);
    log << "Training started at: " << start_time << "\n";
    log << "Training finished at: " << finish_time << "\n";
    log << "Elapsed total time: " << elapsed_seconds << " seconds\n";
    log << "Training status: " << (status == 0 ? "success" : "failed") << "\n";
    log << "Validation status: " << (val_status == 0 ? "success" : "failed") << "\n";
    log << "Model validated: " << validation_model.string() << "\n\n";

    // Ask YOLO to provide its own summary as well	
    return val_status == 0 ? 0 : 1;
}
