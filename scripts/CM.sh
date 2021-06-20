path='../raw_data/CollegeMsg.txt'
datacol='source target time'
filename='CM'
numperm=100
normalize=1
numbase=100
conv=1
directed=1
frenode=1
threshold=36
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

jobname=filename$filename.norm$normalize.conv$conv.frenode$frenode.th$threshold.cc$cc.ft$filter
echo $jobname

nohup  python -u ../main.py -dp $path -l $datacol -fn ../TTP/$filename.txt -np $numperm -n $normalize -nb $numbase -conv $conv -dr $directed -fre $frenode -th $threshold -cc $cc -ft $filter > $jobname.log &
# python ../main.py -dp $path -l $datacol -fn ../TTP/$filename.txt -np $numperm -n $normalize -nb $numbase -conv $conv -dr $directed -fre $frenode -th $threshold -cc $cc -ft $filter