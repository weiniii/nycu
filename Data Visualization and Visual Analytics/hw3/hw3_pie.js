(function (d3) {
    'use strict';
    const dropdownMenu = (selection, props) => {
        const{
          options, onOptionClicked, selectedOption
        } = props;

        let select = selection.selectAll('select').data([null]);
        select = select.enter().append('select')
          .merge(select)
                .on('change', function() {
                  onOptionClicked(this.value);
                });
    
        const option = select.selectAll('option').data(options);
        option.enter().append('option')
            .merge(option)
                .attr('value', d => d)
                .property('selected', d => d === selectedOption)
                .text(d => d);
      };
    
    const onXcolumnClicked = column =>{
        xColumns = column;
        render();
    };

    d3.select('#home').on('click', function(){return window.location.href='311554038.html'});
    
    

    function drawPie(result){
        d3.selectAll('g').remove();
        
        const svg = d3.select('svg');
        const width = svg.attr('width');
        const height = svg.attr('height');
        const margin = 100;

        d3.select('text')
            .text(function(){
                if(xColumns=='popularity'){return 'Popularity : A value between 0 and 100, more high more popular.(1~0 means 10~20) '}
                if(xColumns=='duration_ms'){return 'Duration_ms : The track length in mininute. (0~1 means 0~1 mininute)'}
                if(xColumns=='explicit'){return 'Explicit : Whether or not the track has explicit lyrics. (FALSE means NO)'}
                if(xColumns=='danceability'){return 'Danceability : Describes how suitable a track is for dancing based on a combination of musical elements. (1 is the most)'}
                if(xColumns=='energy'){return 'Energy : Measure of intensity and activity between 0 and 1. (For example, death metal has high energy)'}
                if(xColumns=='key'){return 'Key : The key the track is in.'}
                if(xColumns=='loudness'){return 'Loudness : The overall loudness of a track in decibels (dB).'}
                if(xColumns=='mode'){return 'Mode : Mode indicates the modality (major or minor) of a track'}
                if(xColumns=='speechiness'){return 'Speechiness : Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording the closer to 1'}
                if(xColumns=='acousticness'){return 'Acousticness : A confidence measure from 0.0 to 1.0 of whether the track is acoustic.(1 means closer to acoustic)'}
                if(xColumns=='instrumentalness'){return 'Instrumentalness : Predicts whether a track contains no vocals.(The closer the instrumentalness value is to 1.)'}
                if(xColumns=='liveness'){return 'Liveness : Detects the presence of an audience in the recording.(A value above 0.8 provides strong likelihood that the track is live)'}
                if(xColumns=='valence'){return 'Valence :  A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.'}
                if(xColumns=='tempo'){return 'Tempo : The overall estimated tempo of a track in beats per minute (BPM).'}
                if(xColumns=='time_signature'){return 'Time_signature : An estimated time signature.(The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4)'}
            })
        svg.append('g')
            .attr('class', 'slices')
            .attr('transform', `translate(${width/2},${height/2})`);

        svg.append('g')
            .attr('class', 'labels');
        
        svg.append('g')
            .attr('class', 'lines');
        
        const color = d3.scaleOrdinal()
                        .range(['#f54242', '#f58742', '#f5ce42', '#c2f542',
                                '#42f54b', '#42f5ef', '#42b3f5', '#428af5', 
                                '#4542f5', '#9c42f5', '#d742f5', '#f54269']);

        const total = d3.sum(result, d => d.data)
        
        result.forEach(d => {
            d.percentage = Math.round(((d.data)*100/total))
        })
        const radius = Math.min(width, height)/2 - margin;               
        
        const piechart = d3.pie().value(d => d.percentage)
                            .sort(function(a,b){
                                return d3.ascending(a.percentage, b.percentage)
                            });

        const arc = d3.arc()
                        .innerRadius(0)
                        .outerRadius(radius)
                        .padAngle(0);

        const data_ready = piechart(result);

        const cutePie = svg.select('.slices')
                            .selectAll('g')
                            .data(data_ready)
                            .enter()
                                .append('g')
                                .attr('class', 'arc');
        
        cutePie.append('path')
                .attr('d', arc)
                .attr('fill', function(d){ return(color(d.data.item))})
                .attr('stroke', 'black')
                .style('stroke-width', '2px')
                .style('opacity', 1);

        const inner = 1.5;
        const out = 2.5;

        const keyText = cutePie
            .append('text')
            .attr('transform', d => `translate(${arc.centroid(d)[0] * inner},${arc.centroid(d)[1] * inner})`)
            .text(d => `${d.data.percentage}` + '%')
            .style('text-anchor', 'middle')
            .style('font-size', 16)
            .style('fill', 'black');

        const itemText = cutePie
            .append('text')
            .attr('transform', d => `translate(${arc.centroid(d)[0] * out},${arc.centroid(d)[1] * out})`)
            .text(d => `${d.data.item}`)
            .style('text-anchor', 'middle')
            .style('font-size', 16)
            .style('fill', 'black');

        cutePie.append("line")
            .attr("stroke", "black")
            .attr("x1", function(d){ return arc.centroid(d)[0] * 2 })
            .attr("x2", function(d){ return arc.centroid(d)[0] * 2.1 })
            .attr("y1", function(d){ return arc.centroid(d)[1] * 2 })
            .attr("y2", function(d){ return arc.centroid(d)[1] * 2.1 })

        d3.select('#x-menus')
            .call(dropdownMenu, {
            options: data.columns.slice(5, 20),
            onOptionClicked: onXcolumnClicked,
            selectedOption: xColumns
        })
    }

    let xColumns;
    let keydata;
    let data;
    let where = 'http://vis.lab.djosix.com:2020/data/spotify_tracks.csv';
    d3.csv(where).then(load =>{
        data = load;
        xColumns = 'popularity';
        render();
    });
    const render = () =>{
        if(xColumns=='popularity'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.popularity)
            })
            keydata = [
                {item:'0~1', data: d3.filter(subkeydata,d=>d<10).length},
                {item:'1~2', data: d3.filter(subkeydata,d=>d>=10 & d<20).length},
                {item:'2~3', data: d3.filter(subkeydata,d=>d>=20 & d<30).length},
                {item:'3~4', data: d3.filter(subkeydata,d=>d>=30 & d<40).length},
                {item:'4~5', data: d3.filter(subkeydata,d=>d>=40 & d<50).length},
                {item:'5~6', data: d3.filter(subkeydata,d=>d>=50 & d<60).length},
                {item:'6~7', data: d3.filter(subkeydata,d=>d>=60 & d<70).length},
                {item:'7~8', data: d3.filter(subkeydata,d=>d>=70 & d<80).length},
                {item:'> 8', data: d3.filter(subkeydata,d=>d>=80).length}]
        }
        if(xColumns=='duration_ms'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.duration_ms)
            })
            keydata = [
                {item:'0~1', data: d3.filter(subkeydata,d=>d<60000).length},
                {item:'1~2', data: d3.filter(subkeydata,d=>d>=60000 & d<120000).length},
                {item:'2~3', data: d3.filter(subkeydata,d=>d>=120000 & d<180000).length},
                {item:'3~4', data: d3.filter(subkeydata,d=>d>=180000 & d<240000).length},
                {item:'4~5', data: d3.filter(subkeydata,d=>d>=240000 & d<300000).length},
                {item:'5~6', data: d3.filter(subkeydata,d=>d>=300000 & d<360000).length},
                {item:'> 6', data: d3.filter(subkeydata,d=>d>=360000).length}]
        }
        if(xColumns == 'explicit'){
            var subkeydata = data.map(function(d){
                if(d.explicit=='True'){ return 1}
                else{return 0}
            })
            
            keydata = [
                {item:'FALSE', data: d3.filter(subkeydata,d=>d==0).length},
                {item:'TRUE', data: d3.filter(subkeydata,d=>d==1).length}
            ]
            
        }
        if(xColumns=='danceability'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.danceability)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'0.33~0.66', data: d3.filter(subkeydata,d=> d>=0.33 & d<0.66).length},
                {item:'> 0.66', data: d3.filter(subkeydata,d=>d>=0.66).length}]
        }
        if(xColumns=='energy'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.energy)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'0.33~0.66', data: d3.filter(subkeydata,d=> d>=0.33 & d<0.66).length},
                {item:'> 0.66', data: d3.filter(subkeydata,d=>d>=0.66).length}]
        }
        if(xColumns=='key'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.key)});
            keydata = [
                {item:'C', data: d3.filter(subkeydata,d=>d==0).length},
                {item:'C#', data: d3.filter(subkeydata,d=>d==1).length},
                {item:'D', data: d3.filter(subkeydata,d=>d==2).length},
                {item:'D#', data: d3.filter(subkeydata,d=>d==3).length},
                {item:'E', data: d3.filter(subkeydata,d=>d==4).length},
                {item:'E#', data: d3.filter(subkeydata,d=>d==5).length},
                {item:'F', data: d3.filter(subkeydata,d=>d==6).length},
                {item:'F#', data: d3.filter(subkeydata,d=>d==7).length},
                {item:'G', data: d3.filter(subkeydata,d=>d==8).length},
                {item:'G#', data: d3.filter(subkeydata,d=>d==9).length},
                {item:'A', data: d3.filter(subkeydata,d=>d==10).length},
                {item:'A#', data: d3.filter(subkeydata,d=>d==11).length}]
        }
        if(xColumns=='loudness'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.loudness)
            })
            keydata = [
                {item:'< -20', data: d3.filter(subkeydata,d=>d<-20).length},
                {item:'-20 ~ -10', data: d3.filter(subkeydata,d=>d>=-20 & d<-10).length},
                {item:'> -10', data: d3.filter(subkeydata,d=>d>=-10 & d<0).length}]
        }
        if(xColumns=='mode'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.mode)
            })
            keydata = [
                {item:'Minor Scale', data: d3.filter(subkeydata,d=>d==0).length},
                {item:'Major Scale', data: d3.filter(subkeydata,d=>d==1).length}]
        }
        if(xColumns=='speechiness'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.speechiness)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'> 0.33', data: d3.filter(subkeydata,d=>d>=0.33).length}]
        }
        if(xColumns=='acousticness'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.acousticness)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'> 0.33', data: d3.filter(subkeydata,d=>d>=0.33).length}]
        }
        if(xColumns=='instrumentalness'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.instrumentalness)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'> 0.33', data: d3.filter(subkeydata,d=>d>=0.33).length}]
        }
        if(xColumns=='liveness'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.liveness)
            })
            keydata = [
                {item:'< 0.8', data: d3.filter(subkeydata,d=>d<0.8).length},
                {item:'> 0.8', data: d3.filter(subkeydata,d=>d>=0.8).length}]
        }
        if(xColumns=='valence'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.valence)
            })
            keydata = [
                {item:'< 0.33', data: d3.filter(subkeydata,d=>d<0.33).length},
                {item:'0.33~0.66', data: d3.filter(subkeydata,d=> d>=0.33 & d<0.66).length},
                {item:'> 0.66', data: d3.filter(subkeydata,d=>d>=0.66).length}]
        }
        if(xColumns=='tempo'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.tempo)
            })
            keydata = [
                {item:'< 100', data: d3.filter(subkeydata,d=>d<100).length},
                {item:'100~150', data: d3.filter(subkeydata,d=> d>=100 & d<150).length},
                {item:'150~200', data: d3.filter(subkeydata,d=> d>=150 & d<200).length},
                {item:'> 200', data: d3.filter(subkeydata,d=>d>=200).length}]
        }
        if(xColumns=='time_signature'){
            var subkeydata = data.map(function(d){
                return parseFloat(d.time_signature)});
            keydata = [
                {item:'!= 4', data: d3.filter(subkeydata,d=>d!=4).length},
                {item:'4', data: d3.filter(subkeydata,d=>d==4).length}]
        }







        drawPie(keydata)
    }

}(d3));