import urllib
import zipfile
import nottingham_util
import rnn

url = "www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip"
urllib.urlretrieve(url, "dataset.zip")

zip = zipfile.ZipFile(r'dataset.zip')
zip.extractall('data') 

#sequence learning -- classifying music into two components
#melody and harmony

# ordinary neural nets can't do this, they accept fixed size input
# use RNN  -- special type

#Long Short Term Memory Network

nottingham_util.xreate_model()

rnn.train_model()