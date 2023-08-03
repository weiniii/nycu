(function (d3) {
    'use strict';
    d3.select('#home').on('click', function(){return window.location.href='311554038.html'});
    
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
  
    const onYcolumnClicked = column =>{
        yColumns = column;
        render();
    };

    function scatter(total, data, xColumns, yColumns, resultart){

        d3.selectAll('g').remove();

        const text = svg.append('g');
        text.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .text(yColumns)
            .attr('font-size', '2em')
            .attr('transform', `rotate(-90)translate(-${innerHeight/2 + margin.top}, ${margin.left/2})`);
        
        text.append('text')
            .attr('x', 0)
            .attr('y', 0)
            .text(xColumns)
            .attr('font-size', '2em')
            .attr('transform', `translate(${innerWidth/2 + margin.left}, ${height - margin.bottom/2})`);

        const xdata = data.map(function(d){return parseFloat(d[xColumns])})
        const ydata = data.map(function(d){return parseFloat(d[yColumns])})

        const xScale = d3.scaleLinear()
                        .domain(d3.extent(xdata))
                        .range([0, innerWidth]).nice();

        const yScale = d3.scaleLinear()
                        .domain(d3.extent(ydata))
                        .range([innerHeight, 0]).nice();

        const xAxis = d3.axisBottom(xScale)
                        .tickSize(-innerHeight)
                        .tickPadding(10);
        const yAxis = d3.axisLeft(yScale)
                        .tickSize(-innerWidth)
                        .tickPadding(30);
        
        svg.append('g')
            .call(xAxis)
            .attr('font-size', '13px')
            .attr('class', 'axis')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${margin.left }, ${height - margin.bottom})`);

        svg.append('g')
            .call(yAxis)
            .attr('font-size', '13px')
            .attr('class', 'axis')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${margin.left }, ${margin.top})`);

        svg.append('g')
            .selectAll('circle')
            .data(data)
            .enter()
              .append('circle')
              .attr('fill', d=>{
                if (d.artists==artist1) {return 'red'}
                if (d.artists==artist2){return 'green'}
              })
              .attr('cx', d => xScale(d[xColumns]))
              .attr('cy', d => yScale(d[yColumns]))
              .attr('transform', `translate(${margin.left }, ${margin.top})`)
              .attr('r', 5)
              .on('mouseover', function(d,i){
                d3.select('svg').append('g').append('rect')
                    .attr('width', d3.max([i.track_name.length, i.artists.length])*8)
                    .attr('height', 44)
                    .attr('id', 'mouserect')
                    .attr('x', xScale(i[xColumns]))
                    .attr('y', yScale(i[yColumns]))
                    .attr('fill', function(){
                        if (i.artists==artist1) {return '#fc9090'}
                        if (i.artists==artist2){return '#99fc90'}
                      })
                    .attr('transform', 
                    `translate(${margin.left - d3.max([i.track_name.length, i.artists.length])*(8)/2}, ${-14})`);

                d3.select('svg').append('g').append('text')
                    .attr('id', 'mousetxt')
                    .attr('x', xScale(i[xColumns]))
                    .attr('y', yScale(i[yColumns]))
                    .attr('transform', `translate(${margin.left}, ${8})`)
                    .text(i.artists)
                    .attr('font-size','12px')
                    .attr('fill', 'black');

                d3.select('svg').append('g').append('text')
                    .attr('id', 'mousetxt')
                    .attr('x', xScale(i[xColumns]))
                    .attr('y', yScale(i[yColumns]))
                    .attr('transform', `translate(${margin.left}, ${22})`)
                    .text(i.track_name)
                    .attr('font-size','12px')
                    .attr('fill', 'black');
              })
              .on('mouseleave', function(d){
                d3.selectAll('#mousetxt').remove()
                d3.selectAll('#mouserect').remove()
              });

        d3.select('#x-menus')
            .call(dropdownMenu, {
            options: total.columns.slice(5, 6).concat(total.columns.slice(8, 20)),
            onOptionClicked: onXcolumnClicked,
            selectedOption: xColumns
        })
        d3.select('#y-menus').call(dropdownMenu, {
            options: total.columns.slice(5, 6).concat(total.columns.slice(8, 20)),
            onOptionClicked: onYcolumnClicked,
            selectedOption: yColumns
        });
        d3.select('#datalistOptions1').selectAll('option')
            .data(resultart)
            .enter()
            .append('option') 
            .attr('value', function (d) { return d; });
        d3.select('#exampleDataList1')
            .on('change', function(){datalist1onclicked(this.value)})
            .on('search', function(){ artist1 = null});

        d3.select('#datalistOptions2').selectAll('option')
            .data(resultart) 
            .enter()
            .append('option') 
            .attr('value', function (d) { return d; });
        d3.select('#exampleDataList2')
            .on('change', function(){datalist2onclicked(this.value)})
            .on('search', function(){ artist2 = null});

        const datalist1onclicked = d =>{
            artist1 = d
            render()
        }
        const datalist2onclicked = d =>{
            artist2 = d
            render()
        }
    };
    
    const svg = d3.select('svg');
    const width =+ svg.attr('width');
    const height =+ svg.attr('height');
    const margin = {
        left : 120,
        right : 90,
        top :40,
        bottom : 120,
    };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    let data;
    let total;
    let xColumns;
    let yColumns;
    let resultart;
    let artist1 = 'Gen Hoshino';
    let artist2 = 'Ben Woodward';

    let where = 'http://vis.lab.djosix.com:2020/data/spotify_tracks.csv';
    d3.csv(where).then(load =>{
        total = load;
        
        var art = [];
        let t = total.length;
        for(let i =0 ; i<t ; i++){
            art[i] = total[i].artists;
        }
        resultart = new Set();
        var repeat = new Set();
        art.forEach(item => {
            resultart.has(item) ? repeat.add(item) : resultart.add(item);
        })
        resultart = Array.from(resultart);

        xColumns = total.columns[5];
        yColumns = total.columns[9];
        var adata = total.filter(function(element){   
            return element.artists == artist1 | element.artists == artist2
        });
        data = adata

        render();
    });

    const render = () =>{
        var adata = total.filter(function(element){   
            return element.artists == artist1 | element.artists == artist2
        });
        data = adata
        scatter(total, data, xColumns, yColumns, resultart)
    }
    
}(d3));