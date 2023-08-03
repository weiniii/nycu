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

  const  scatterPlot = (selection, props) => {
    const {
    	title,
      xvalue,
     	xLabel,
     	yvalue,
     	yLabel,
    	circleRadius,
     	margin,
      width,
      height,
      data
    } = props;

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const xScale = d3.scaleLinear()
    	.domain(d3.extent(data, xvalue))
    	.range([0, innerWidth])
    	.nice();

    const yScale = d3.scaleLinear()
    	.domain(d3.extent(data, yvalue))
    	.range([innerHeight, 0])
    	.nice();

    const g = selection.selectAll('.container').data([null]);
    const gEnter = g
    	.enter().append('g')
    		.attr('class', 'container');
    gEnter.merge(g)
    		.attr('transform', `translate(${margin.left}, ${margin.top})`);

    const xAxis = d3.axisBottom(xScale)
    	.tickSize(-innerHeight)
    	.tickPadding(10);

    const yAxis = d3.axisLeft(yScale)
    	.tickSize(-innerWidth)
    	.tickPadding(10);

    const yAxisG = g.select('.y-axis');
    const yAxisGEnter = gEnter
    	.append('g')
  			.attr('class', 'y-axis');
    yAxisG
    	.merge(yAxisGEnter)
    		.call(yAxis)
  			.selectAll('.domain').remove();

    const yAxisLabelText = yAxisGEnter
    	.append('text')
        .attr('class', 'axis-label')
        .attr('y', -80)
        .attr('text-anchor','middle')
        .attr('fill', 'black')
        .attr('transform', `rotate(-90)`)
    	.merge(yAxisG.select('.axis-label'))
    		.attr('x', -innerHeight/2)
        .text(yLabel);

    const xAxisG = g.select('.x-axis');
    const xAxisGEnter = gEnter
    	.append('g')
  			.attr('class', 'x-axis');
    xAxisG
    	.merge(xAxisGEnter)
    		.attr('transform', `translate(0, ${innerHeight})`)
    		.call(xAxis)
  			.selectAll('.domain').remove();

    const xAxisLabelText = xAxisGEnter
    	.append('text')
        .attr('class', 'axis-label')
        .attr('y', 80)
        .attr('fill', 'black')
    	.merge(xAxisG.select('.axis-label'))
    		.attr('x', innerWidth/2)
        .text(xLabel);

    const circles = g.merge(gEnter)
    	.selectAll('circle').data(data);
    circles
      .enter().append('circle')
    		.attr('fill', d => {
      		if(d['class'] === 'Iris-setosa'){return 'green'
          }else {
          	if(d['class'] === 'Iris-versicolor'){return '#f03232'
            }else {
            	return '#1a0ac7'
            }}})
    		.attr('cx', innerWidth/2)
    		.attr('cy', innerHeight/2)
    		.attr('r', 0)
    	.merge(circles)
    	.transition().duration(1000)
    	.delay((d, i) => i * 10)
    		.attr('cy', d => yScale(yvalue(d)))
    		.attr('cx', d => xScale(xvalue(d)))
    		.attr('r',circleRadius);

  };

  const svg = d3.select('svg');
  const width = +svg.attr('width');
  const height = +svg.attr('height');

  const textrect = svg.append('g')
		.attr('transform',`translate(0, 10)`);
  textrect
    .transition().duration(2000)
      .attr('transform',`translate(0, 50)`)

  const textrect1 = textrect
    .append('rect')
      .attr('x', 850 )
      .attr('width', 30)
      .attr('height', 30)
      .attr('fill','green');

  svg.append("text")
    .attr("x", 885)
    .attr("y", 20 )
    .style("font-size", "1em")
    .text("setosa")
  .transition().duration(2000)
      .attr('transform',`translate(0, 50)`);

  const textrect2 = textrect
  .append('rect')
    .attr('x', 850 )
    .attr('y', 50 )
    .attr('width', 30)
    .attr('height', 30)
    .attr('fill','#f03232');

  svg.append("text")
    .attr("x", 885)
    .attr("y", 70 )
    .style("font-size", "1em")
    .text("versicolor")
  .transition().duration(2000)
    .attr('transform',`translate(0, 50)`);;

  const textrect3 = textrect
  .append('rect')
    .attr('x', 850 )
    .attr('y', 100 )
    .attr('width', 30)
    .attr('height', 30)
    .attr('fill','#1a0ac7');

  svg.append("text")
    .attr("x", 885)
    .attr("y", 120 )
    .style("font-size", "1em")
    .text("virginica")
  .transition().duration(2000)
    .attr('transform',`translate(0, 50)`);

  let data;
  let xColumn;
  let yColumn;

  const onXcolumnClicked = column =>{
  	xColumn = column;
    render();
  };

  const onYcolumnClicked = column =>{
  	yColumn = column;
    render();
  };

  const render = () =>{

    d3.select('#x-menus')
      .call(dropdownMenu, {
        options: data.columns.slice(0, 4),
        onOptionClicked: onXcolumnClicked,
        selectedOption: xColumn
      })
    	.attr('x',100);
    d3.select('#y-menus').call(dropdownMenu, {
      options: data.columns.slice(0, 4),
      onOptionClicked: onYcolumnClicked,
      selectedOption: yColumn
    });
    svg.call(scatterPlot, {
      xvalue: d => d[xColumn],
    	xLabel: xColumn,
     	yvalue: d => d[yColumn],
    	yLabel: yColumn,
    	circleRadius: 5,
  		margin: {top: 10, right: 140, bottom: 120, left: 120},
      width,
      height,
      data
    });
  };

  d3.csv('http://vis.lab.djosix.com:2020/data/iris.csv').then(loadeddata =>{

    data = loadeddata;

    var result=data.filter(function(element, index){
      return index !=150 ;
    });
    data = result;
    data.columns = loadeddata.columns;

    xColumn = data.columns[0];
    yColumn = data.columns[1];
    render();
  });

}(d3));