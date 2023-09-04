for dir in /work/u1887834/imagenet/train/*; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "$dir has $count files"
    fi
done
