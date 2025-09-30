(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        blue - block
        purple - block
        orange - block
        yellow - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on blue green)
        (on purple blue)
        (on orange purple)
        (on yellow orange)
        (on red yellow)
        (clear red)
        ; green -> blue -> purple -> orange -> yellow -> red
    )
    (:goal (and (on yellow green)
        (on blue purple)
        (on orange red)))
)
    