import gdown

# a file
# url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
# output = "fcn8s_from_caffe.npz"
url = "https://drive.google.com/file/d/1-0NLNOgtzbYOOmzZ9gv1rozTP3TG3_9K/view?usp=share_link"
output = "file.cvs"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
