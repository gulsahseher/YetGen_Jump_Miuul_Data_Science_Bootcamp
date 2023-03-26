############################################################
# Sayılar(Numbers) ve Karakter Dizileri (Strings)
############################################################

print("Hello World") #string
print("Hello AI Era") #string
9.2 #float
9 #integer

type(9.2) # veri yapısının ne olduğunu söyler

############################################################
# Atamalar ve Değişkenler (Assigments & Variables)
############################################################

a = 9
b = "hello ai era"
c = 10
a*c

d = a-c

##########################################################################
# Virtual Enviroment (Sanal Ortam) ve (Package Management) Paket Yönetimi
##########################################################################

# Sanal ortamların listelenmesi : conda env list
# Sanal ortam oluşturma : conda create -n myenv
# Sanal ortamı silme : conda env remove -n myenv
# Sanal ortamı aktif etme : conda activate myenv
# Sanal ortamdan çıkma : conda deactivate
# Yüklü paketlerin listelenmesi : conda list
# Paket yükleme : conda install numpy
# Aynı anda birden fazla paket yükleme : conda install numpy scipy pandas
# Paket silme : conda remove package_name
# Paket yükseltme : conda upgrade conda
# Tüm paketlerin yükseltilmesi : conda upgrade -all
# .yaml'daki bilgilerin alınarak sanal ortamın oluşturulması : conda env create -f environment.yaml
#

# pip: pypi (python package index) paket yönetim aracı
# Paket yükleme : pip install packet_name
# Versiyona göre paket yükleme : pip install pandas==1.5.2
