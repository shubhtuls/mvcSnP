blenderExec=$1
blendFile=$2
objpath=$3
pngpath=$4
az=$5
el=$6
dist=$7
upsamp=$8
theta=$9
$blenderExec $blendFile --background --python ../renderer/renderPose.py -- $objpath $pngpath $az $el $dist $upsamp $theta > /dev/null