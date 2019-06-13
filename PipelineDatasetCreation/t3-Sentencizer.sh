JAVA_HOME=/nfs/misc/literature/tools/java/jre1.8.0_45
#JAVA_HOME=/nfs/misc/literature/yangx/jdk-10.0.1
rpath="/nfs/misc/literature/lit-textmining-pipelines"

start=`date +%s`

UKPMCXX=$rpath/lib
OTHERS=$UKPMCXX/ebitmjimenotools.jar:$UKPMCXX/monq.jar:$UKPMCXX/mallet.jar:$UKPMCXX/mallet-deps.jar:$UKPMCXX/marie.jar:$UKPMCXX/pipeline180822_notOA.jar:$UKPMCXX/commons-lang-2.4.jar:$UKPMCXX/ojdbc6-11.1.0.7.0.jar:$UKPMCXX/ie.jar:$UKPMCXX/commons-io-2.0.1.jar:$UKPMCXX/jopt-simple-3.2.jar

xmlDir=$1 # change the path of this xmlDir
title=$'###   3/6 - SENTENCIZER   ###\n'
echo "$title"
cd ${xmlDir}
pwd

for fname in PMC*; do
if [[ ${fname} == *"-fulltxt"* ]];then
    fname2=${fname:0:-12}
    echo $fname2
    cat ${fname} | sed 's/"article-type=/" article-type=/' | perl ${rpath}/bin/SectionTagger_XML_inline.perl | $JAVA_HOME/bin/java -cp $OTHERS:$rpath/lib/pmcxslpipe20170801.jar ebi.ukpmc.xslpipe.Pipeline -stdpipe -stageSpotText | $JAVA_HOME/bin/java -cp $OTHERS:$rpath/lib/pmcxslpipe20170801.jar ebi.ukpmc.xslpipe.Pipeline -stdpipe -outerText | $JAVA_HOME/bin/java -cp $OTHERS:${rpath}/lib/Sentenciser160415.jar ebi.ukpmc.sentenciser.Sentencise -rs '<article[^>]+>' -ok -ie UTF-8 -oe UTF-8 | $JAVA_HOME/bin/java -cp $OTHERS:${rpath}/lib/Sentenciser160415.jar ebi.ukpmc.sentenciser.SentCleaner -stdpipe > ${fname2}-sentencized.xml
fi
done
end=`date +%s`
runtime=$((end-start))
duration=$'\nDuration'
echo "$duration : $runtime sec"
end_tool=$'\nSentencizer DONE\n'
echo "$end_tool"