const hostApi = process.env.NODE_ENV === "development" ? "http://localhost" : "https://sing-generator-node.herokuapp.com";
const portApi = process.env.NODE_ENV === "development" ? 8080 : "";
const baseURLApi = `${hostApi}${portApi ? `:${portApi}` : ``}/api`;
const redirectUrl = process.env.NODE_ENV === "development" ? "http://localhost:5000/light-blue-react" : "https://demo.flatlogic.com/light-blue-react";
global.XMLHttpRequest = require("xhr2");

const url = "http://172.16.120.19:5000";
//const url = "http://43.200.170.131:5000";

function httpGet(theURL) {
  const xmlHttp = new XMLHttpRequest();
  console.log(url.concat(theURL));
  xmlHttp.open("GET", url.concat(theURL), false);
  xmlHttp.send(null);
  return JSON.parse(xmlHttp.responseText);
}

export default {
  httpGet,
  redirectUrl,
  hostApi,
  portApi,
  baseURLApi,
  remote: "https://sing-generator-node.herokuapp.com",
  isBackend: process.env.REACT_APP_BACKEND,
  auth: {
    email: 'admin@flatlogic.com',
    password: 'password'
  },
  app: {
    colors: {
      dark: "#333964",
      light: "#0A0417",
      sea: "#4A4657",
      sky: "#3A3847",
      rain: "#3846AA",
      middle: "#3390C3"
    },
  }
};