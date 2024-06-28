## Start Jekyll Server
```shell
cd docs
jekyll serve --livereload
```

##### Alternative
```sh
jekyll serve --port 4001
```

## Close an existing server
```shell
sudo lsof -i :4000
sudo kill <process_id>
```
