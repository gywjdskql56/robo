import React, { PureComponent } from 'react';
import PropTypes from 'prop-types';
import cx from 'classnames';
import { Table, Button } from 'reactstrap';

import s from './TableContainer.module.scss';

const states = {
  cash: 'primary',
  bond: 'success',
  stock: 'danger',
};

class TableContainer extends PureComponent {
  static propTypes = {
    data: PropTypes.arrayOf(
      PropTypes.shape({
        ticker: PropTypes.string,
        name: PropTypes.string,
        unit: PropTypes.string,
        price: PropTypes.string,
        date: PropTypes.string,
        sector: PropTypes.string,
        asset_class: PropTypes.string,
      }),
    ).isRequired,
  }

  render() {
    const { data } = this.props;
    const keys = Object.keys(data[0]).map(i => i.toUpperCase());
    keys.shift(); // delete "id" key
    return (
      <Table className={cx('mb-0', s.table)} borderless responsive>
        <thead>
          <tr className="text-white">
            {keys.map((key, index) => (
              index === 0
              ? <th key={key} scope="col" className={"p-3"}><span className="ps-2">{key}</span></th>
              : <th key={key} scope="col" className={"p-3"}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {
            data.map(({ ticker, name, unit, price, date, sector, asset_class }) => (
              <tr key={ticker}>
                <td className="p-3">{ticker}</td>
                <td>{name}</td>
                <td>{unit}</td>
                <td>{price}</td>
                <td>{date}</td>
                <td>{sector}</td>
                <td>
                  <Button
                    color={states[asset_class.toLowerCase()]}
                    size="xs"
                    className="px-2"
                  >
                    {asset_class}
                  </Button>
                </td>
              </tr>
            ))
          }
        </tbody>
      </Table>
    );
  }
}

export default TableContainer;
