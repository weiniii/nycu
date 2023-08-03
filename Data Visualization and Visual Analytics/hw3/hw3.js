(function (d3) {
    'use strict';
    
    const svg = d3.select('svg');
    const height =+ svg.attr('height')
    const width =+ svg.attr('width')
    
    const circle = d3.select('svg').append('circle');
    circle
        .attr('cx', width/2)
        .attr('cy', height/2)
        .attr('r', 200)
        .attr('fill', '#5bf08f');

    const spotify = d3.select('svg').append('text').attr('class', 'spotify');
    spotify
        .attr('x', width/2)
        .attr('y', height/2)
        .text('Spotify')
        .on('click', function(){return window.location.href='311554038.html'});

    const hint = d3.select('svg').append('text').attr('class', 'hint');
    hint
        .attr('x', width/2)
        .attr('y', height/2 + 100)
        .text('Data Analyze by Wei')

    const pie = d3.select('svg').append('text').attr('class', 'pie');
    pie
        .attr('x', width/4)
        .attr('y', 3*height/4 + 100)
        .text('Pie Chart')
        .on('click', function(){return window.location.href='311554038_pie.html'});
    
    const scatter = d3.select('svg').append('text').attr('class', 'scatter');
    scatter
        .attr('x', 3*width/4)
        .attr('y', 3*height/4 + 100)
        .text('Scatter Plot')
        .on('click', function(){return window.location.href='311554038_scatter.html'});

    const tip = d3.select('svg').append('text').attr('class', 'tip');
    tip
        .attr('x', width/2)
        .attr('y', 3*height/4 + 150)
        .text('Click Pie Chart or Scatter Plot will go to the own website')

    const warn = d3.select('svg').append('text').attr('class', 'warn');
    warn
        .attr('x', width/2)
        .attr('y', 3*height/4 + 170)
        .text('Going to scatter-website will wait about twenty seconds!!!');

}(d3));