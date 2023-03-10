import Highcharts from 'highcharts';
import config from './config';
const colors = config.chartColors;

let columnColors = [colors.blue, colors.green, colors.orange, colors.red, colors.default, colors.gray, colors.teal, colors.pink];
let lineColors = [colors.blue, colors.green, colors.orange];
const port = {
    "Conservative":{
        "goal_prob":"-",
        "loss_prob":"-",
        "return":"-",
        "sec_weight_df":"-",
        "class_weight_df":[
      {
        "name": "Stock",
        "value": 17.37
      },
      {
        "name": "Bond",
        "value": 74.08
      },
      {
        "name": "Alt",
        "value": 8.55
      }
    ],
        "wealth_path_idx":"-",
        "wealth_path":"-",
        "wealth_path_goal":"-",
        "wealth_path_goal_step":"-",
    },
    "Neutral":{
    },
    "Aggressive":{
    }
}
sessionStorage.setItem("init_port", JSON.stringify(port))
sessionStorage.setItem("risktype","Conservative")
sessionStorage.setItem("value", 3);
export const chartData = {
  apex: {
    column: {
      series: [{
        data: [21, 22, 10, 28, 16, 21, 13, 30]
      }],
      options: {
        chart: {
          height: 350,
          type: 'bar'
        },
        legend: {
          show: true,
          labels: {
            colors: colors.textColor,
          },
          itemMargin: {
            horizontal: 10,
            vertical: 5
          },
        },
        colors: columnColors,
        plotOptions: {
          bar: {
            columnWidth: '45%',
            distributed: true
          }
        },
        dataLabels: {
          enabled: false,
        },
        xaxis: {
          categories: ['John', 'Joe', 'Jake', 'Amber', 'Peter', 'Mary', 'David', 'Lily'],
          labels: {
            style: {
              colors: columnColors,
              fontSize: '14px'
            }
          },
          axisBorder: {
            show: false
          },
          axisTicks: {
            show: false
          }
        },
        yaxis: {
          labels: {
            style: {
              color: colors.textColor,
            }
          }
        },
        tooltip: {
          theme: 'dark'
        },
        grid: {
          borderColor: colors.gridLineColor
        }
      }
    },
    pie: {
      series: [25, 15, 44, 55, 41, 17],
      options: {
        labels: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        theme: {
          monochrome: {
            enabled: true,
            color: colors.blue,
          }
        },
        stroke: {
          show: false,
          width: 0
        },
        legend: false,
        responsive: [{
          breakpoint: 480,
          options: {
            chart: {
              width: 200
            },
            legend: {
              position: 'bottom'
            }
          }
        }]
      }
    }
  },
  echarts: {
    line: {
      color: lineColors,
      tooltip: {
        trigger: 'none',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['2015 Precipitation', '2016 Precipitation'],
        textStyle: {
          color: colors.textColor
        }
      },
      grid: {
        top: 70,
        bottom: 50,
      },
      xAxis: [
        {
          type: 'category',
          axisTick: {
            alignWithLabel: true
          },
          axisLine: {
            onZero: false,
            lineStyle: {
              color: lineColors[1]
            }
          },
          axisPointer: {
            label: {
              formatter: function (params) {
                return '기간별 목표수익률:  ' + params.value
                  + (params.seriesData.length ? '：' + params.seriesData[0].data : '');
              }
            }
          },
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].wealth_path_idx
        },

        {
          type: 'category',
          axisTick: {
            alignWithLabel: true
          },
          axisLine: {
            onZero: false,
            lineStyle: {
              color: lineColors[0]
            }
          },
          axisPointer: {
            label: {
              formatter: function (params) {
                return '예상 포트수익률  ' + params.value
                  + (params.seriesData.length ? '：' + params.seriesData[0].data : '');
              }
            }
          },
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].wealth_path_idx
        }
      ],
      yAxis: [
        {
          type: 'value',
          axisLabel: {
            color: colors.textColor
          },
          axisLine: {
            lineStyle: {
              color: colors.textColor
            }
          },
          splitLine: {
            lineStyle: {
              color: colors.gridLineColor
            }
          },
          axisPointer: {
            label: {
              color: colors.dark
            }
          }
        }
      ],
      series: [
        {
          name: '목표 자산 수준',
          type: 'line',
          smooth: true,
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].wealth_path_goal
        },
        {
          name: '예상 자산 추이',
          type: 'line',
          xAxisIndex: 1,
          smooth: true,
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].wealth_path
        },

        {
          name: '기간별 목표 자산 수준',
          type: 'line',
          smooth: true,
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].wealth_path_goal_step
        },
      ]
    },
    donut: {
      tooltip: {
        trigger: 'item',
        formatter: "{a} <br/>{b}: {c} ({d}%)"
      },
      legend: {
        show: false
      },
      color: [colors.blue, colors.green, colors.orange, colors.red, colors.purple],
      series: [
        {
          name: 'Access source',
          type: 'pie',
          radius: ['50%', '70%'],
          avoidLabelOverlap: false,
          label: {
            normal: {
              show: false,
              position: 'center'
            },
            emphasis: {
              show: true,
              textStyle: {
                fontSize: '30',
                fontWeight: 'bold'
              }
            }
          },
          labelLine: {
            normal: {
              show: false
            }
          },
          data: [
            {value: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[0].value, name: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[0].name},
            {value: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[1].value, name: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[1].name},
            {value: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[2].value, name: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].class_weight_df[2].name},
          ]
        }
      ]
    },
    river: {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'line',
          lineStyle: {
            color: 'rgba(0,0,0,0.2)',
            width: 1,
            type: 'solid'
          }
        }
      },

      legend: {
        data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].sec_weight_tickers,
        textStyle: {
          color: colors.textColor
        }
      },
      color: [colors.default, colors.lawngreen, colors.blanchedalmond, colors.blue, colors.green, colors.orange, colors.red, colors.purple, colors.gray, colors.dark, colors.teal, colors.pink, colors.red, colors.Magenta, colors.firebrick],
      singleAxis: {
        top: 50,
        bottom: 50,
        axisTick: {},
        axisLabel: {
          color: colors.textColor
        },
        type: 'time',
        axisPointer: {
          animation: true,
          label: {
            show: true,
            color: colors.dark
          }
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: [colors.gridLineColor],
            type: 'dashed',
            opacity: 0.2
          }
        },
        axisLine: {
          lineStyle: {
            color: colors.textColor
          }
        },
      },

      series: [
        {
          type: 'themeRiver',
          itemStyle: {
            emphasis: {
              shadowBlur: 20,
              shadowColor: 'rgba(0, 0, 0, 0.8)'
            }
          },
          data: JSON.parse(sessionStorage.getItem("init_port"))[sessionStorage.getItem("risktype")].sec_weight_df
        }
      ]
    }
  },
  highcharts: {
    mixed: {
      chart: {
        type: 'spline',
        height: 350,
        backgroundColor: 'transparent'
      },
      exporting: {
        enabled: false
      },
      title: {
        text: 'Snow depth at Vikjafjellet, Norway',
        style: {
          color: colors.textColor
        }
      },
      credits: {
        enabled: false
      },
      xAxis: {
        type: 'datetime',
        dateTimeLabelFormats: { // don't display the dummy year
          month: '%e. %b',
          year: '%b'
        },
        labels: {
          style: {
            color: colors.textColor
          }
        }
      },
      yAxis: {
        min: 0,
        title: {
          enabled: false
        },
        labels: {
          style: {
            color: colors.textColor
          }
        },
        gridLineColor: colors.gridLineColor
      },
      tooltip: {
        headerFormat: '<b>{series.name}</b><br>',
        pointFormat: '{point.x:%e. %b}: {point.y:.2f} m'
      },
      legend: {
        enabled: false
      },
      plotOptions: {
        series: {
          marker: {
            enabled: false,
            symbol: 'circle'
          }
        }
      },
      colors: [colors.green, colors.blue, colors.red],

      series: [{
        name: "Winter 2014-2015",
        data: [
          [Date.UTC(1970, 10, 25), 0],
          [Date.UTC(1970, 11, 6), 0.25],
          [Date.UTC(1970, 11, 20), 1.41],
          [Date.UTC(1970, 11, 25), 1.64],
          [Date.UTC(1971, 0, 4), 1.6],
          [Date.UTC(1971, 0, 17), 2.55],
          [Date.UTC(1971, 0, 24), 2.62],
          [Date.UTC(1971, 1, 4), 2.5],
          [Date.UTC(1971, 1, 14), 2.42],
          [Date.UTC(1971, 2, 6), 2.74],
          [Date.UTC(1971, 2, 14), 2.62],
          [Date.UTC(1971, 2, 24), 2.6],
          [Date.UTC(1971, 3, 1), 2.81],
          [Date.UTC(1971, 3, 11), 2.63],
          [Date.UTC(1971, 3, 27), 2.77],
          [Date.UTC(1971, 4, 4), 2.68],
          [Date.UTC(1971, 4, 9), 2.56],
          [Date.UTC(1971, 4, 14), 2.39],
          [Date.UTC(1971, 4, 19), 2.3],
          [Date.UTC(1971, 5, 4), 2],
          [Date.UTC(1971, 5, 9), 1.85],
          [Date.UTC(1971, 5, 14), 1.49],
          [Date.UTC(1971, 5, 19), 1.27],
          [Date.UTC(1971, 5, 24), 0.99],
          [Date.UTC(1971, 5, 29), 0.67],
          [Date.UTC(1971, 6, 3), 0.18],
          [Date.UTC(1971, 6, 4), 0]
        ]
      }, {
        name: "Winter 2015-2016",
        type: 'areaspline',
        data: [
          [Date.UTC(1970, 10, 9), 0],
          [Date.UTC(1970, 10, 15), 0.23],
          [Date.UTC(1970, 10, 20), 0.25],
          [Date.UTC(1970, 10, 25), 0.23],
          [Date.UTC(1970, 10, 30), 0.39],
          [Date.UTC(1970, 11, 5), 0.41],
          [Date.UTC(1970, 11, 10), 0.59],
          [Date.UTC(1970, 11, 15), 0.73],
          [Date.UTC(1970, 11, 20), 0.41],
          [Date.UTC(1970, 11, 25), 1.07],
          [Date.UTC(1970, 11, 30), 0.88],
          [Date.UTC(1971, 0, 5), 0.85],
          [Date.UTC(1971, 0, 11), 0.89],
          [Date.UTC(1971, 0, 17), 1.04],
          [Date.UTC(1971, 0, 20), 1.02],
          [Date.UTC(1971, 0, 25), 1.03],
          [Date.UTC(1971, 0, 30), 1.39],
          [Date.UTC(1971, 1, 5), 1.77],
          [Date.UTC(1971, 1, 26), 2.12],
          [Date.UTC(1971, 3, 19), 2.1],
          [Date.UTC(1971, 4, 9), 1.7],
          [Date.UTC(1971, 4, 29), 0.85],
          [Date.UTC(1971, 5, 7), 0]
        ]
      }, {
        name: "Winter 2016-2017",
        type: 'areaspline',
        data: [
          [Date.UTC(1970, 9, 15), 0],
          [Date.UTC(1970, 9, 31), 0.09],
          [Date.UTC(1970, 10, 7), 0.17],
          [Date.UTC(1970, 10, 10), 0.1],
          [Date.UTC(1970, 11, 10), 0.1],
          [Date.UTC(1970, 11, 13), 0.1],
          [Date.UTC(1970, 11, 16), 0.11],
          [Date.UTC(1970, 11, 19), 0.11],
          [Date.UTC(1970, 11, 22), 0.08],
          [Date.UTC(1970, 11, 25), 0.23],
          [Date.UTC(1970, 11, 28), 0.37],
          [Date.UTC(1971, 0, 16), 0.68],
          [Date.UTC(1971, 0, 19), 0.55],
          [Date.UTC(1971, 0, 22), 0.4],
          [Date.UTC(1971, 0, 25), 0.4],
          [Date.UTC(1971, 0, 28), 0.37],
          [Date.UTC(1971, 0, 31), 0.43],
          [Date.UTC(1971, 1, 4), 0.42],
          [Date.UTC(1971, 1, 7), 0.39],
          [Date.UTC(1971, 1, 10), 0.39],
          [Date.UTC(1971, 1, 13), 0.39],
          [Date.UTC(1971, 1, 16), 0.39],
          [Date.UTC(1971, 1, 19), 0.35],
          [Date.UTC(1971, 1, 22), 0.45],
          [Date.UTC(1971, 1, 25), 0.62],
          [Date.UTC(1971, 1, 28), 0.68],
          [Date.UTC(1971, 2, 4), 0.68],
          [Date.UTC(1971, 2, 7), 0.65],
          [Date.UTC(1971, 2, 10), 0.65],
          [Date.UTC(1971, 2, 13), 0.75],
          [Date.UTC(1971, 2, 16), 0.86],
          [Date.UTC(1971, 2, 19), 1.14],
          [Date.UTC(1971, 2, 22), 1.2],
          [Date.UTC(1971, 2, 25), 1.27],
          [Date.UTC(1971, 2, 27), 1.12],
          [Date.UTC(1971, 2, 30), 0.98],
          [Date.UTC(1971, 3, 3), 0.85],
          [Date.UTC(1971, 3, 6), 1.04],
          [Date.UTC(1971, 3, 9), 0.92],
          [Date.UTC(1971, 3, 12), 0.96],
          [Date.UTC(1971, 3, 15), 0.94],
          [Date.UTC(1971, 3, 18), 0.99],
          [Date.UTC(1971, 3, 21), 0.96],
          [Date.UTC(1971, 3, 24), 1.15],
          [Date.UTC(1971, 3, 27), 1.18],
          [Date.UTC(1971, 3, 30), 1.12],
          [Date.UTC(1971, 4, 3), 1.06],
          [Date.UTC(1971, 4, 6), 0.96],
          [Date.UTC(1971, 4, 9), 0.87],
          [Date.UTC(1971, 4, 12), 0.88],
          [Date.UTC(1971, 4, 15), 0.79],
          [Date.UTC(1971, 4, 18), 0.54],
          [Date.UTC(1971, 4, 21), 0.34],
          [Date.UTC(1971, 4, 25), 0]
        ]
      }]
    },
  }
};

export let liveChartInterval = null;

export const liveChart = {
  liveChartInterval: null,
  colors: [colors.blue],
  chart: {
    backgroundColor: 'transparent',
    height: 170,
    type: 'spline',
    animation: Highcharts.svg, // don't animate in old IE
    marginRight: 10,
    events: {
      load: function () {

        // set up the updating of the chart each second
        var series = this.series[0];
        liveChartInterval = setInterval(function () {
          var x = (new Date()).getTime(), // current time
            y = Math.random();
          series.addPoint([x, y], true, true);
        }, 1000);
      }
    }
  },

  time: {
    useUTC: false
  },
  credits: {
    enabled: false
  },
  title: false,
  xAxis: {
    type: 'datetime',
    tickPixelInterval: 150,
    labels: {
      style: {
        color: colors.textColor
      }
    },
    lineWidth: 0,
    tickWidth: 0
  },
  yAxis: {
    title: {
      enabled: false
    },
    plotLines: [{
      value: 0,
      width: 1,
      color: colors.textColor
    }],
    labels: {
      style: {
        color: colors.textColor
      }
    },
    gridLineColor: colors.gridLineColor
  },
  tooltip: {
    headerFormat: '<b>{series.name}</b><br/>',
    pointFormat: '{point.x:%Y-%m-%d %H:%M:%S}<br/>{point.y:.2f}'
  },
  legend: {
    enabled: false
  },
  exporting: {
    enabled: false
  },
  series: [{
    name: 'Random data',
    data: (function () {
      // generate an array of random data
      var data = [],
        time = (new Date()).getTime(),
        i;

      for (i = -19; i <= 0; i += 1) {
        data.push({
          x: time + i * 1000,
          y: Math.random()
        });
      }
      return data;
    }())
  }]
};