(function (d3) {
  'use strict';

  const sideContent = (selection, colorMap) => {
    
    const initialX = 750;
    const initialY = 70;
    
    const paddingX = parseInt(25);
    const paddingY = parseInt(40);
    const alignText = parseInt(6);
    
    const g = selection;
    const gSide = g.append('g')
    	.attr('transform', `translate(${initialX}, ${initialY})`);
    
    const cat1 = 'Iris-setosa';
    const cat2 = 'Iris-versicolor';
    const cat3 = 'Iris-virginica';
    
    gSide.append('rect')
  		.attr('width', 20)
    	.attr('height', 8)
    	.attr('fill', colorMap[cat1]);

    gSide.append('text')
    	.attr('x', paddingX)
    	.attr('y', alignText)
    	.text(cat1);
    
    gSide.append('rect')
  		.attr('width', 20)
    	.attr('height', 8)
    	.attr('y', paddingY)
    	.attr('fill', colorMap[cat2]);
    
    gSide.append('text')
    	.attr('x', paddingX)
    	.attr('y', String(alignText+paddingY))
    	.text(cat2);
    
    gSide.append('rect')
  		.attr('width', 20)
    	.attr('height', 8)
    	.attr('y', String(2*paddingY))
    	.attr('fill', colorMap[cat3]);
    
    gSide.append('text')
    	.attr('x', paddingX)
    	.attr('y', String(alignText+2*paddingY))
    	.text(cat3);
  };

  /*sepal length,sepal width,petal length,petal width,class*/

  const svg = d3.select('svg');
  const width = +svg.attr('width');
  const height = +svg.attr('height');

  const column = ['sepal length', 'sepal width', 'petal length', 'petal width'];

  const colorMap = {
  	'Iris-setosa' : 'green',
  	'Iris-versicolor' : '#f03232',
    'Iris-virginica' : '#1a0ac7'
  };

  const render = data => {
  	
    //畫布設定
    const margin = { top: 90, right: 180, bottom: 90, left: 60 };
    const innerHeight = height - margin.top - margin.bottom;
    const innerWidth = width - margin.right - margin.left;
    
    const title = 'Iris Parallel Coordinate Plot';
    
    const g = svg.append('g')
    	.attr('transform', `translate(${margin.left}, ${margin.top})`);

    const x = d3.scalePoint()
     	.domain(column)
     	.range([0, innerWidth]);
    
    const y = d3.scaleLinear()
      	.domain([0, 8])
      	.range([innerHeight, 0]);
    
    const axisY = d3.axisLeft().scale(y)
    	.tickPadding(5)
    	.tickSize(5);
    
    const generateLine = d => 
  		  d3.line()(column.map(p => [x(p), y(d[p])] ));
    
    const generateLineMoving = d =>
    	d3.line()(column.map(p => [correctPos(p), y(d[p])] ));
    
    
    //資料
    const pathG = g.selectAll('path').data(data).enter()
    	.append('path')
    	.attr('d', d => generateLine(d))
    	.attr('stroke',  d => colorMap[d.class])
    	.attr('fill', 'none')
    	.attr('opacity', 0.35)
    	.attr('stroke-width', 2);
    
    
    //軸
    const axisG = g.selectAll('.axes').data(column).enter()
    	.append('g')
    		.attr('class', 'axes')
    		.each(function(d) { d3.select(this).call(axisY);})
    		.attr('transform', d => 'translate(' + x(d) +',0)');
    
    var position = {};
    
    const correctPos = d =>{
  		return position[d] == null ? x(d) : position[d];
    };
    
    
    const dragged = (d,xDragPosition) => {
      position[d] = x(d);
      
      position[d] = Math.min(innerWidth+30, Math.max(-30, xDragPosition));
      column.sort( (p,q) => correctPos(p) - correctPos(q));
      
    	x.domain(column);
    	pathG.attr('d', d => generateLineMoving(d));
    	axisG.attr('transform', d => 'translate(' + correctPos(d) +',0)');
    }; 

    
    const dragended = d => {
    	 delete position[d];
    	 transition(pathG).attr("d",  d => generateLine(d));
    	 transition(axisG).attr("transform", p => "translate(" + x(p) + ",0)"); 
    };
    
    //drag
    axisG.call(d3.drag()
              .subject(function(d) { return {x: x(d)}; })
              .on('drag', (d,event) => dragged(d, d3.event.x))
              .on('end', d => dragended(d))
    );
    
    axisG
      .on('mouseover', d => d3.select(undefined).style("cursor", "wait"))
      .on('mouseout' , d => d3.select(undefined).style("cursor", "default"));
    
    
    //axis name
    axisG.append('text')
    		.attr('fill','black')
    		.attr('transform', `translate(0,${innerHeight})`)
    		.attr('y', 30)
    		.attr('text-anchor', 'middle')
    		.attr('font-size', 18)
    		.text(d => d);
    
    //icon
    g.call(sideContent, colorMap);
    
    //title
    g.append('text')
    	.attr('class', 'title')
    	.attr('x', innerWidth/2)
    	.attr('y', '-15')
    	.style('text-anchor', 'middle')
    	.text(title);
    
     
  };

  const transition = g =>  
      g.transition().duration(350);
    

  d3.csv('http://vis.lab.djosix.com:2020/data/iris.csv').then( data => {
   		data.forEach(d => {
        //str to int
      	d['sepal length'] = +d['sepal length'];
        d['sepal width'] = +d['sepal width'];
        d['petalLength'] = +d['petal length'];
        d['petalWidth'] = +d['petal width'];
      });
    render(data);
  });

}(d3));