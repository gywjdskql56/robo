const chartColors = {
  blue: '#2477ff',
  green: '#2d8515',
  orange: '#e49400',
  red: '#db2a34',
  purple: '#474D84',
  dark: '#040620',
  teal: '#4179cf',
  pink: '#e671b8',
  gray: '#d6dee5',
  default: '#595d78',
  textColor: '#e0e0e1',
  gridLineColor: '#040620'
};
global.XMLHttpRequest = require("xhr2");

const url = "http://172.16.120.19:5001";
//const url = "http://43.200.170.131:5000";
export function httpGet(theURL) {
  const xmlHttp = new XMLHttpRequest();
  console.log(url.concat(theURL));
  xmlHttp.open("GET", url.concat(theURL), false);
  xmlHttp.send(null);
  return JSON.parse(xmlHttp.responseText);
}

export default {
  chartColors,
}
