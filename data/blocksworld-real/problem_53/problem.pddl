(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        blue - block
        green - block
        yellow - block
        purple - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on blue orange)
        (clear blue)
        (ontable green)
        (on yellow green)
        (clear yellow)
        (ontable purple)
        (on red purple)
        (clear red)
        ; orange -> blue
        ; green -> yellow
        ; purple -> red
    )
    (:goal (and (on orange yellow)
        (on green blue)
        (on purple red)))
)
    