#!/bin/bash
echo 'Count file lines'
echo "find ./data_model -name 'webtext_thread_*.txt' | xargs wc -l | sort"
echo 'Locate from a list of files if a filename contains keyword'
echo 'tail data_model/webtext_thread_*.txt | egrep "webtext_thread|{your-keyword}"'