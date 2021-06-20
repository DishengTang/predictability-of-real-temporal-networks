path='../raw_data/realityMining.txt'
datacol='source target weight time'
filename='RM'
numperm=100
normalize=1
numbase=100
conv=1
directed=0
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

jobname=filename$filename.norm$normalize.conv$conv.frenode$frenode.cc$cc.ft$filter
echo $jobname

nohup  python -u ../main.py -dp $path -l $datacol -fn ../TTP/$filename.txt -np $numperm -n $normalize -nb $numbase -conv $conv -dr $directed -fre $frenode -cc $cc -ft $filter > $jobname.log &
# python ../main.py -dp $path -l $datacol -fn ../TTP/$filename.txt -np $numperm -n $normalize -nb $numbase -conv $conv -dr $directed -fre $frenode -cc $cc -ft $filter