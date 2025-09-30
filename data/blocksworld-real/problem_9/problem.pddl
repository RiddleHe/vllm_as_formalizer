(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        purple - block
        red - block
        green - block
        yellow - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on purple orange)
        (on red purple)
        (clear red)
        (ontable green)
        (on yellow green)
        (on blue yellow)
        (clear blue)
        ; orange -> purple -> red
        ; green -> yellow -> blue
    )
    (:goal (and (on orange blue)
        (on red green)
        (on yellow purple)))
)
    