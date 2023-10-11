import React from "react";
import { Nav, NavLink, NavMenu }
    from "./NavbarElements";
 
const Navbar = () => {
    return (
        <>
            <Nav>
                <NavMenu>
                    <NavLink to="/" activeStyle>Home</NavLink>
                    <NavLink to="/about" activeStyle>Learn More About Model and Calculation</NavLink>
                </NavMenu>
            </Nav>
        </>
    );
};
 
export default Navbar;