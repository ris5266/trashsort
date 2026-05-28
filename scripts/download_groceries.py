import os
import urllib.request
import tarfile

# freiburg groceries dataset
URL = "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
TAR = "data/groceries.tar.gz"
OUT = "data/groceries"


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(TAR):
        print("downloading ~516MB ...")
        urllib.request.urlretrieve(URL, TAR)
    print("extracting ...")
    with tarfile.open(TAR, "r:*") as t:
        t.extractall(OUT)
    n = sum(len(fs) for _, _, fs in os.walk(os.path.join(OUT, "images")))
    print("done -> %s (%d images)" % (OUT, n))


if __name__ == "__main__":
    main()
