path='./raw_data/forum.txt'
datacol='source target time'
filename='CF'
conv=1
directed=1
frenode=1
threshold=10
cc=1
filter=1

echo $path
echo $datacol
echo $filename
echo $conv
echo $directed
echo $frenode
echo $threshold
echo $cc
echo $filter

jobname=filename$filename.conv$conv.frenode$frenode.th$threshold.cc$cc.ft$filter
echo $jobname

nohup  python -u main.py -dp $path -l $datacol -fn ./TTP/$filename.txt -conv $conv -dr $directed -fre $frenode -th $threshold -cc $cc -ft $filter > $jobname.log &
# python main.py -dp $path -l $datacol -fn ./TTP/$filename.txt -conv $conv -dr $directed -fre $frenode -th $threshold -cc $cc -ft $filter