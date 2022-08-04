# python-project-template

## Commands used

- Show image
```
$ python3 main.py program=show_image image_idx=0
```

- Corregistrate both cameras:
```
$ python3 main.py program=corregistrate image_idx=0
```

- Select signatures
```
$ python3 main.py program=select_signatures image_idx=0 selector.item=background
```
```
$ python3 main.py program=select_signatures image_idx=0 selector.item=object
```

- Segmentation
```
$ python3 main.py program=test_segmentation image_idx=0
```
```
$ python3 main.py program=perform_segmentation image_idx=0
```

- Prepare dataset
```
$ python3 main.py program=prepare_data
```

- Create signatures
```
$ python3 main.py program=create_signatures
```

- Visualise signatures
```
$ python3 main.py program=visualise_signatures
```


