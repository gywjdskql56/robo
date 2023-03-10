export default {
  serverOverview: [
    {
      data: [{
        data: [4, 6, 5, 7, 5],
      }],
      width: '100%',
      height: 30,
      options: {
        stroke: {
          width: 1
        },
        markers: {
          size: 4,
          colors: '#57B955',
          shape: "circle",
          strokeWidth: 0,
          hover: {
            size: 5,
            colors: '#fff',
          }
        },
        colors: [
          '#4E85BD'
        ],
        grid: {
          padding: {
            left: 10,
            right: 10,
            top: 10,
            bottom: 10
          }
        }
      }
    },
    {
      data: [{
        data: [2, 3, 1, 4, 4],
      }],
      width: '100%',
      height: 30,
      options: {
        stroke: {
          width: 1
        },
        markers: {
          size: 4,
          colors: '#57B955',
          shape: "circle",
          strokeWidth: 0,
          hover: {
            size: 5,
            colors: '#fff',
          }
        },
        colors: [
          '#4E85BD'
        ],
        grid: {
          padding: {
            left: 10,
            right: 10,
            top: 10,
            bottom: 10
          }
        }
      }
    }
  ],
  tasks: [
    {
      id: 0,
      type: '경제 지표 발표',
      title: '미시간 대학 소비심리지표',
      time: '(한국시간) 3:00',
    },
    {
      id: 1,
      type: '금리 회의',
      title: 'FOMC',
      time: '(한국시간) 11:00',
    },
    {
      id: 2,
      type: '경제 지표 발표',
      title: 'CPI',
      time: '(한국시간) 15:00',
    },
    {
      id: 3,
      type: '실적 발표',
      title: 'JP 모건',
      time: '(한국시간) 23:00',
    },
  ],
  bigStat: [
    {
      product: '실제 자산',
      total: '4,232',
      color: 'primary',
      registrations: {
        value: 83,
        profit: true,
      },
      bounce: {
        value: 4.5,
        profit: true,
      },
    },
    {
      product: '목표 자산',
      total: '754',
      color: 'success',
      registrations: {
        value: 30,
        profit: true,
      },
      bounce: {
        value: 2.5,
        profit: true,
      },
    },
    {
      product: '괴리액',
      total: '1,025',
      color: 'danger',
      registrations: {
        value: 230,
        profit: true,
      },
      bounce: {
        value: 21.5,
        profit: false,
      },
    },
  ],
  notifications: [
    {
      id: 0,
      icon: 'bell',
      color: 'danger',
      content: '골드만삭스 호실적에 1.2% 상승',
    },
    {
      id: 0,
      icon: 'bell',
      color: 'danger',
      content: 'JP모건 호실적에 2.5% 상승',
    },
    {
      id: 0,
      icon: 'bell',
      color: 'danger',
      content: '웰스파고 실적부진에도 3.3% 상승',
    },
    {
      id: 1,
      icon: 'bell',
      color: 'info',
      content: '테슬라 가격 인하 소식에 4.% 급락',
    },
    {
      id: 2,
      icon: 'bell',
      color: 'info',
      content: '골드만삭스 경고에 방산주 급학',
    }
  ],
  table: [
    {
      id: 0,
      ticker: 'DRIV US EQUITY',
      name: 'ISHARES SELF-DRIVING EV&TECH',
      unit: 'USD',
      price: '$25',
      date: '2022.12.01',
      sector: 'TECH',
      asset_class: 'STOCK',
    },
    {
      id: 1,
      ticker: 'PBD US EQUITY',
      name: 'Clean Energy',
      unit: 'USD',
      price: '$1.7',
      date: '2022.12.01',
      sector: 'UTILITY',
      asset_class: 'STOCK',
    },
    {
      id: 2,
      ticker: 'SIMS US EQUITY',
      name: 'SPDR S&P KENSHO INTELLIGENT',
      unit: 'USD',
      price: '$15.67',
      date: '2022.12.01',
      sector: 'REITS',
      asset_class: 'STOCK',
    },
    {
      id: 3,
      ticker: 'AGG US EQUITY',
      name: 'ISHARES CORE U.S. AGGREGATE',
      unit: 'USD',
      price: '$24.5',
      date: '2022.12.01',
      sector: 'BOND',
      asset_class: 'BOND',
    },
    {
      id: 4,
      ticker: 'USD',
      name: 'USD',
      unit: 'USD',
      price: '$94.7',
      date: '2022.12.01',
      sector: 'CASH',
      asset_class: 'CASH',
    },
  ],
  backendData: {
    visits: {
      count: '5,323,000원',
      logins: '3,000원',
      sign_out_pct: 0.5,
      rate_pct: 4.5
    },
    performance: {
      sdk: {
        this_period_pct: 100,
        last_period_pct: 30,
      },
      integration: {
        this_period_pct: 27,
        last_period_pct: 30,
      }
    },
    server: {
      1: {
        pct: 60,
        temp: 37,
        frequency: 3.3
      },
      2: {
        pct: 54,
        temp: 31,
        frequency: 3.3
      }
    },
    revenue: getRevenueData(),
    mainChart: getMainChartData()
  }
};

function getRevenueData() {
  const data = [];
  const seriesCount = 3;
  const accessories = ['채권', '유동', '주식'];

  for (let i = 0; i < seriesCount; i += 1) {
    data.push({
      label: accessories[i],
      data: Math.floor(Math.random() * 100) + 1,
    });
  }

  return data;
}

function getMainChartData() {
  function generateRandomPicks(minPoint, maxPoint, picksAmount, xMax) {
    let x = 0;
    let y = 0;
    const result = [];
    const xStep = 1;
    const smoothness = 0.3;
    const pointsPerPick = Math.ceil(xMax / ((picksAmount * 2) + 1) / 2);

    const maxValues = [];
    const minValues = [];

    for (let i = 0; i < picksAmount; i += 1) {
      const minResult = minPoint + Math.random();
      const maxResult = maxPoint - Math.random();

      minValues.push(minResult);
      maxValues.push(maxResult);
    }

    let localMax = maxValues.shift(0);
    let localMin = 0;
    let yStep = parseFloat(((localMax - localMin) / pointsPerPick).toFixed(2));

    for (let j = 0; j < Math.ceil(xMax); j += 1) {
      result.push([x, y]);

      if ((y + yStep >= localMax) || (y + yStep <= localMin)) {
        y += yStep * smoothness;
      } else if ((result[result.length - 1][1] === localMax) || (result[result.length - 1][1] === localMin)) {
        y += yStep * smoothness;
      } else {
        y += yStep;
      }

      if (y > localMax) {
        y = localMax;
      } else if (y < localMin) {
        y = localMin;
      }

      if (y === localMin) {
        localMax = maxValues.shift(0) || localMax;

        const share = (localMax - localMin) / localMax;
        const p = share > 0.5 ? Math.round(pointsPerPick * 1.2) : Math.round(pointsPerPick * share);

        yStep = parseFloat(((localMax - localMin) / p).toFixed(2));
        yStep *= Math.abs(yStep);
      }

      if (y === localMax) {
        localMin = minValues.shift(0) || localMin;

        const share = (localMax - localMin) / localMax;
        const p = share > 0.5 ? Math.round(pointsPerPick * 1.5) : Math.round(pointsPerPick * 0.5);

        yStep = parseFloat(((localMax - localMin) / p).toFixed(2));
        yStep *= -1;
      }

      x += xStep;
    }

    return result;
  }

  const d1 = generateRandomPicks(0.2, 3, 4, 90);
  const d2 = generateRandomPicks(0.4, 3.8, 4, 90);
  const d3 = generateRandomPicks(0.2, 4.2, 3, 90);

  return [d1, d2, d3];
}

