Use Commands
==================

Before use, make sure you successfully performed the setup of the application following [setup guide](setup_guide.md).

The application is currently configured in a way that a user interacts with it via a command line interface. The app is used utilizing the following commands:

```
$ python3 main.py
```
when run locally and
```
$ run
```
when run in a docker container. Use both with `--help` flag to display help information.

For use, run the command in the following format:
```
$ run [OPTIONS] program=COMMAND
```

, where available options are:

```
  Options:
    --help, -h     Application's help
    --cfg,  -c     Show config (add arg: 'job', 'hydra', 'all' -> example: 'run --cfg job')
    --info, -i     Print Hydra information

  Commands:
    check_images            Check the initial stage of raw images
    show_image              Show image defined by image_idx (default: image_idx=0)
    select_signatures       Select spectral signatures from a specific part of the image
    corregistrate           Find corregistration parameters between two cameras
    test_segmentation       Performs test segmentation (dry run)
    perform_segmentation    Performs segmentation with save options
    prepare_data            Prepares data as set in config files
    create_signatures       Create signatures as set in config files
    visualise_signatures    Visualise signatures as set in config files

  Complementary commands:
    image_idx       Index of image at which program should start
```


Other options are written in config files and can be changed depending on the configuration. Which configs will be used during execution is selected in `configs/custom.yaml` file.
```
  Config files:
    camera1/camera.yaml
    camera2/camera.yaml
    data_loader/data_loader.yaml
    selector/selector.yaml
    segmentator/segmentator.yaml
    preparator/preparator.yaml
    visualiser/visualiser.yaml
```

Settings in `configs` folder can be edited before execution of the program by directly changing the parameters in `.yaml` files. Additionally, these can be dynamically modified during execution

For example, you can set the name of the item for which you are using selector:

* for object:
```
$ run program=select_signatures selector.item=object
```
 * for background:
```
$ run program=select_signatures selector.item=background
```
