#!/bin/bash

Create the .streamlit directory in the home folder

mkdir -p ~/.streamlit/

Write the config content using a Here Document (cat << EOF)

cat << EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
EOF