path='./raw_data/manufacturingEmails.txt'
datacol='source target weight time'
filename='ME'
conv=1
directed=1
frenode=0
cc=0
filter=1

echo $path
echo $datacol
echo $filename
echo $conv
echo $directed
echo $frenode
echo $cc
echo $filter

jobname=filename$filename.conv$conv.frenode$frenode.cc$cc.ft$filter
echo $jobname

nohup  python -u main.py -dp $path -l $datacol -fn ./TTP/$filename.txt -conv $conv -dr $directed -fre $frenode -cc $cc -ft $filter > $jobname.log &
# python main.py -dp $path -l $datacol -fn ./TTP/$filename.txt -conv $conv -dr $directed -fre $frenode -cc $cc -ft $filter